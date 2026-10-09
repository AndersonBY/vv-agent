"""ApprovalProvider/Broker transport; the parked operation and inbox own decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vv_agent.approval import ApprovalDecision, ApprovalRequest

from .records import InboxItem, digest

if TYPE_CHECKING:
    from .kernel import _Driver
    from .reducer import Attempt


def resolve_approval(driver: _Driver, attempt: Attempt) -> bool:
    assert attempt.wait is not None
    runtime, plan = driver.runtime, attempt.execution_plan
    assert plan.turn_id is not None
    handle = attempt.wait["handle"]
    provider, broker = runtime.config.approval_provider, runtime.approval_broker
    deadline = attempt.wait["deadline_ms"]
    decision = None
    if broker is not None:
        # Drain only real broker decisions; wait(0) alone fabricates a timeout.
        with broker._condition:
            if handle["request_id"] in broker._decisions:
                decision = broker.wait(handle["request_id"], timeout=0)
    if decision is None and deadline is not None and driver.scope.poll(driver.store).db_now_ms >= deadline:
        decision = ApprovalDecision.timeout("Approval request timed out.")
        if broker is not None:
            broker.discard(handle["request_id"])
    if decision is None and provider is not None and broker is not None:
        if broker.pending_request(handle["request_id"]) is not None:
            return False
        executor = runtime.functions.orchestrator._resolve_executor(plan._payload["request"]["name"])
        assert executor is not None
        request = ApprovalRequest(
            request_id=handle["request_id"],
            tool_name=plan._payload["request"]["name"],
            tool_call_id=plan._payload["request"]["id"],
            arguments=plan.payload["request"]["arguments"],
            run_id=plan.turn_id or "",
            trace_id=plan.turn_id or "",
            agent_name=runtime.agent.name,
            cycle_index=int(plan._payload["dependencies"][0].rsplit("/", 1)[1]),
            metadata={
                "tool_metadata": dict(executor.metadata),
                "session_id": driver.sid,
                "timeout_seconds": max(0, (deadline - driver.scope.poll(driver.store).db_now_ms) / 1000)
                if deadline is not None
                else None,
            },
        )
        if not provider.should_request(request):
            decision = ApprovalDecision.allow()
        else:
            broker.register(request)
            try:
                decision = provider.decide(request)
            except Exception as exc:
                broker.discard(request.request_id)
                driver.close("failed", "agent_failed", str(exc))
                return True
            if decision is not None:
                broker.resolve(request.request_id, decision)
                decision = broker.wait(request.request_id, timeout=0)
    if decision is None:
        return False
    payload = {
        "operation_id": plan.operation_id,
        "attempt": plan.attempt,
        "request_id": handle["request_id"],
        "request_digest": handle["request_digest"],
        "scope": handle["scope"],
        "decision": "approve" if decision.action == "allow" else decision.action,
        "reason": decision.reason,
        "metadata": decision.metadata,
    }
    answer = InboxItem(
        f"approval/{plan.operation_id}/{plan.attempt}/answer/{digest(payload)}",
        "approval_answer",
        payload,
        plan.turn_id,
        driver.state.turns[plan.turn_id].start._payload["generation"],
    )
    with driver.store.atomic() as tx:
        tx.push(driver.sid, answer)
    return True
