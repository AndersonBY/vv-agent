"""Provider registration and recovery through the real App Server producers."""

import json
import os
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import monotonic
from typing import Any, cast

import pytest

from vv_agent import Agent, RunBudgetLimits, RunConfig, ScriptedModelProvider, ToolPolicy
from vv_agent.run_config import effective_run_config
from vv_agent.session.kernel import drive
from vv_agent.session.records import InboxItem
from vv_agent.tools.registry import ToolRegistry
from vv_agent.types import LLMResponse, ToolCall, ToolExecutionResult

from .conftest import open_store

PRODUCER = runpy.run_path(str(Path(__file__).parents[2] / "scripts/app_server_provider_fixtures.py"))
THREAD, TURN = PRODUCER["THREAD"], PRODUCER["TURN"]
BusinessDB, JobsHost, JobsProvider, Server = (PRODUCER[name] for name in ("BusinessDB", "JobsHost", "JobsProvider", "Server"))
produce = PRODUCER["produce"]
business_tool = PRODUCER["business_tool"]


@pytest.fixture
def business(tmp_path):
    return BusinessDB(tmp_path / "business.sqlite")


def wait_for(predicate, timeout=20):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        Event().wait(0.02)
    raise AssertionError("producer did not reach the requested boundary")


def test_accept_recreate_resume_wake_and_two_reconnects_use_one_job(store, database, business):
    first = Server(store, business)
    receipt = first.start()
    handle = first.attempt().wait["handle"]
    assert first.attempt().state == "parked"
    assert first.rpc("thread/read", {"threadId": THREAD})["result"]["thread"]["status"] == "provider_running"
    duplicate = first.rpc(
        "turn/start", {"threadId": THREAD, "inputId": "work", "expectedTurnId": None, "input": [{"type": "text", "text": "go"}]}
    )
    assert duplicate["result"]["turnId"] == receipt["turnId"]
    first.resume()
    first.server.wake(THREAD)
    first.join()
    first.close()
    with open_store(database) as reopened:
        second = Server(reopened, business)
        try:
            for identity in ("reconnected-1", "reconnected-2"):
                transport = second.connect(identity)
                second.rpc("thread/resume", {"threadId": THREAD}, transport=transport)
                second.resume(transport=transport)
                second.server.wake(THREAD)
                second.join()
            assert second.attempt().wait["handle"] == handle
            business.finish()
            second.due()
            second.resume()
            assert second.attempt().result.payload["context"] == "normal"
            assert second.attempt().result.payload["result"]["tool_call_id"] == "business-call"
            assert business.count("submit") == business.count("effect") == 1
            assert business.count("query") >= 1 and business.count("sync") == 0
            assert second.host.builds >= 3
        finally:
            second.close()


@pytest.mark.parametrize("invalid", ["call", "status", "evidence", "empty_evidence", "ambiguous", "fields"])
def test_invalid_query_keeps_the_handle_until_valid_result(store, business, invalid):
    server = Server(store, business)
    try:
        server.start()
        handle = server.attempt().wait["handle"]
        business.finish()
        business.setting("invalid", invalid)
        server.due()
        server.resume()
        assert server.attempt().state == "parked"
        assert server.attempt().wait["handle"] == handle
        assert not server.state().turns[TURN].ended
        warnings = server.rpc("thread/read", {"threadId": THREAD})
        assert "result" in warnings
        assert any(w.get("params", {}).get("code") == "provider_result_invalid" for w in server.messages)
        business.setting("invalid", "")
        server.resume()
        assert server.attempt().state == "completed"
        assert business.count("submit") == 1
    finally:
        server.close()


@pytest.mark.parametrize("invalid", ["call", "status", "empty_evidence", "ambiguous", "fields"])
def test_invalid_submit_is_unknown_with_no_automatic_retry(store, business, invalid):
    business.setting("submit_result", True)
    business.setting("invalid", invalid)
    server = Server(store, business)
    try:
        server.start()
        attempt = server.attempt()
        assert attempt.state == "unknown"
        assert attempt.unknown.payload["reason"] == "provider_result_invalid"
        assert attempt.unknown.payload["retry"] == "stop"
        assert attempt.unknown.payload["dispatch_evidence"]
        server.rpc("thread/read", {"threadId": THREAD})
        assert any(w.get("params", {}).get("code") == "provider_result_invalid" for w in server.messages)
        server.server.wake(THREAD)
        server.join()
        assert business.count("submit") == business.count("effect") == 1
    finally:
        server.close()


@pytest.mark.parametrize("field", ["provider", "operation_id", "attempt", "request_digest", "extra"])
def test_invalid_acceptance_cannot_create_a_wait_or_retry(store, business, field):
    business.setting("invalid_handle", field)
    server = Server(store, business)
    try:
        server.start()
        assert server.attempt().state == "unknown"
        assert server.attempt().wait is None
        assert server.attempt().unknown.payload["reason"] == "provider_result_invalid"
        assert server.attempt().unknown.payload["retry"] == "stop"
        assert business.count("submit") == business.count("effect") == 1
    finally:
        server.close()


def test_query_cannot_replace_retained_handle(store, business):
    server = Server(store, business)
    try:
        server.start()
        handle = server.attempt().wait["handle"]
        business.setting("changed_query_handle", True)
        server.due()
        server.resume()
        assert server.attempt().wait["handle"] == handle
        server.rpc("thread/read", {"threadId": THREAD})
        assert any(w.get("params", {}).get("code") == "provider_result_invalid" for w in server.messages)
    finally:
        server.close()


@pytest.mark.parametrize("mode", ["confirmed", "accepted", "unknown", "invalid"])
def test_cancel_from_wire_retains_three_levels_and_replay(store, database, business, mode):
    server = Server(store, business)
    server.start()
    handle = server.attempt().wait["handle"]
    business.setting("cancel", "confirmed" if mode == "invalid" else mode)
    if mode == "invalid":
        business.setting("invalid", "evidence")
    params = {"threadId": THREAD, "expectedTurnId": TURN, "actionId": "cancel"}
    requested = server.rpc("turn/interrupt", params)
    assert "result" in requested
    server.join()
    state = server.state()
    assert state.turns[TURN].ended
    terminal = server.terminal()
    assert terminal["status"] == "cancelled"
    assert bool(terminal["unconfirmed_operations"]) == (mode != "confirmed")
    if mode == "confirmed":
        assert server.attempt().result.payload["context"] == "audit"
    else:
        assert server.attempt().wait["handle"] == handle
    assert "result" in server.rpc("turn/interrupt", params)
    server.join()
    assert business.count("cancel") == 1
    server.close()
    with open_store(database) as reopened:
        second = Server(reopened, business)
        try:
            read = second.rpc("thread/read", {"threadId": THREAD})
            assert "result" in read
            assert second.terminal() == terminal
            second.rpc("thread/events/list", {"threadId": THREAD, "limit": 256})
            assert second.terminal() == terminal
            assert "result" in second.rpc("turn/interrupt", params)
            second.join()
            assert business.count("cancel") == 1
            assert business.count("submit") == business.count("effect") == 1
        finally:
            second.close()


@pytest.mark.parametrize(
    "field",
    ["signature", "provider", "call", "operation", "attempt", "digest", "generation", "missing_generation", "turn", "usage"],
)
def test_forged_callbacks_do_not_complete_or_replace_the_wait(store, business, field):
    server = Server(store, business)
    try:
        server.start()
        handle = server.attempt().wait["handle"]
        business.finish()
        business.setting("query_paused", True)
        provider = JobsProvider(business)
        item = provider.callback(handle, input_id=f"forged-{field}")
        if field == "generation":
            item = replace(item, generation=1)
        elif field == "missing_generation":
            item = replace(item, generation=None)
        elif field == "turn":
            item = replace(item, target_turn_id="wrong-turn")
        else:
            changes = {
                "signature": {"evidence": ["forged-signature"]},
                "provider": {"provider_binding": "impostor"},
                "call": {"result": item.payload["result"] | {"tool_call_id": "WRONG"}},
                "operation": {"operation_id": "wrong-operation"},
                "attempt": {"attempt": 2},
                "digest": {"request_digest": "0" * 64},
                "usage": {"usage": {"business_units": 999}},
            }
            item = replace(item, payload=item.payload | changes[field])
        with store.atomic() as tx:
            tx.push(THREAD, item)
        server.server.wake(THREAD)
        server.join()
        assert server.state().applied_inputs[item.input_id].payload["disposition"] == "rejected"
        assert server.attempt().wait["handle"] == handle
        assert server.attempt().result is None and business.count("submit") == 1
    finally:
        server.close()


def test_authenticated_but_invalid_callback_is_rejected_and_late_result_is_audit(store, business):
    server = Server(store, business)
    try:
        server.start()
        handle = server.attempt().wait["handle"]
        business.finish()
        business.setting("query_paused", True)
        business.setting("invalid", "call")
        invalid = JobsProvider(business).callback(handle, input_id="invalid-trusted-result")
        with store.atomic() as tx:
            tx.push(THREAD, invalid)
        server.server.wake(THREAD)
        server.join()
        applied = server.state().applied_inputs[invalid.input_id].payload
        assert applied["disposition"] == "rejected" and applied["reason"] == "provider_result_invalid"
        assert server.attempt().wait["handle"] == handle
        server.rpc("turn/interrupt", {"threadId": THREAD, "expectedTurnId": TURN, "actionId": "cancel"})
        server.join()
        business.setting("invalid", "")
        late = JobsProvider(business).callback(handle, input_id="late")
        with store.atomic() as tx:
            tx.push(THREAD, late)
        server.server.wake(THREAD)
        server.join()
        assert server.attempt().result.payload["context"] == "audit"
        assert server.state().active_turn_id is None
        assert server.terminal()["status"] == "cancelled"
    finally:
        server.close()


def test_missing_named_binding_never_invokes_function(store, business):
    first = Server(store, business)
    first.start()
    handle = first.attempt().wait["handle"]
    first.close()
    second = Server(store, business, bound=False)
    try:
        second.due()
        second.resume()
        assert second.attempt().wait["handle"] == handle
        assert not second.state().turns[TURN].ended
        second.rpc("thread/read", {"threadId": THREAD})
        assert any(w.get("params", {}).get("code") == "missing_host_binding" for w in second.messages)
        assert business.count("sync") == 0 and business.count("submit") == 1
    finally:
        second.close()


@pytest.mark.parametrize("gate", ["policy", "arguments", "budget", "approval"])
def test_preflight_and_budget_run_before_provider_submission(store, business, gate):
    args = {"value": "not-an-int"} if gate == "arguments" else {"value": 1}
    config = RunConfig(
        model_provider=ScriptedModelProvider.from_steps(
            "test", "m", [LLMResponse("work", [ToolCall("business-call", "work", args)]), LLMResponse("denied safely")]
        ),
        workspace="/fixture",
        tool_registry_factory=ToolRegistry,
        tool_policy=ToolPolicy(can_use_tool=lambda _name, _args: False) if gate == "policy" else None,
        budget_limits=RunBudgetLimits(max_tool_calls=0) if gate == "budget" else None,
    )
    server = Server(store, business, config=config, needs_approval=gate == "approval")
    try:
        server.start(wait=gate != "approval")
        if gate == "approval":
            wait_for(
                lambda: any(
                    a.wait and a.wait["handle"]["kind"] == "approval"
                    for op in server.state().operations.values()
                    for a in op.attempts.values()
                )
            )
            wait = server.attempt().wait["handle"]
            plan = server.attempt().execution_plan
            with store.atomic() as tx:
                tx.push(
                    THREAD,
                    InboxItem(
                        "premature-result",
                        "provider_result",
                        {
                            "operation_id": plan.operation_id,
                            "attempt": plan.attempt,
                            "request_digest": plan.payload["request_digest"],
                            "provider_binding": "work",
                            "result": ToolExecutionResult("business-call", "premature").to_dict(),
                            "evidence": ["premature"],
                            "usage": {},
                        },
                        target_turn_id=TURN,
                        generation=0,
                    ),
                )
            response = server.rpc(
                "approval/resolve", {"threadId": THREAD, "turnId": TURN, "requestId": wait["request_id"], "decision": "deny"}
            )
            assert "result" in response, response
            server.join()
            assert server.state().applied_inputs["premature-result"].payload["disposition"] == "rejected"
        assert business.count("submit") == business.count("effect") == business.count("sync") == 0
    finally:
        server.close()


def test_registration_is_validated_merged_copied_and_conflicts_reject(store, business):
    p1, p2 = JobsProvider(business), JobsProvider(business)
    original = {"work": p1}
    config = RunConfig(tool_providers=original)
    original.clear()
    assert config.tool_providers == {"work": p1}
    merged = effective_run_config(Agent("a", "a"), RunConfig(tool_providers={"work": p2}), runner_defaults=config)
    assert merged.tool_providers == {"work": p2}
    host = JobsHost(business)
    from vv_agent.session.surfaces import SessionDriver

    driver = SessionDriver(store=store)
    try:
        assembled = replace(host.build_run_config(None), tool_providers={"work": p1})
        rt = driver.runtime(host._agent, assembled)
        child = rt.for_agent(host._agent, rt.config)
        assert child.providers == rt.providers and child.providers is not rt.providers
        assembled.tool_providers.clear()
        assert rt.providers == {"work": p1}
        delegated = Agent("parent", "Delegate.", tools=[host._agent.as_tool(name="delegate")])
        with pytest.raises(ValueError, match="conflict"):
            driver.runtime(delegated, replace(host.build_run_config(None), tool_providers={"delegate": p1}))
    finally:
        driver.close()
    for name in ("", " work", "work ", "model", "function", "ask_user", "create_sub_task"):
        with pytest.raises(ValueError):
            RunConfig(tool_providers={name: p1})
    with pytest.raises(TypeError):
        RunConfig(tool_providers={"work": cast(Any, object())})


@pytest.mark.parametrize("denied", [False, True])
def test_child_creation_and_reconstruction_inherit_provider_and_cannot_widen_policy(store, business, denied):
    child = Agent("child", "CHILD-JOB", tools=[business_tool(business)], tool_policy=ToolPolicy(allowed_tools=["work"]))
    parent = Agent("parent", "PARENT-JOB", tools=[child.as_tool(name="delegate", description="Run child")])

    def model(request):
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse("done")
        is_child = any("CHILD-JOB" in (m.content or "") for m in request.messages if m.role == "system")
        return (
            LLMResponse("work", [ToolCall("business-call", "work", {"value": 1})])
            if is_child
            else LLMResponse("delegate", [ToolCall("delegate-call", "delegate", {"task_description": "do job"})])
        )

    config = RunConfig(
        model_provider=ScriptedModelProvider.from_callback("test", "m", model),
        workspace="/fixture",
        tool_registry_factory=ToolRegistry,
        tool_policy=ToolPolicy(disallowed_tools=["work"]) if denied else None,
    )
    server = Server(store, business, config=config, agent=parent)
    try:
        server.start()
        created = [
            row.record
            for sid in store.list_sessions()
            for row in store.read(sid, limit=1).records
            if row.record.payload.get("attributes", {}).get("child_admission")
        ]
        assert len(created) == 1
        sid = created[0].session_id
        rt = server.server.kernel.runtime(parent, server.host.build_run_config(None))
        rebuilt = rt.child_runtime(store, sid)
        assert rebuilt.providers["work"] is rt.providers["work"]
        if denied:
            assert business.count("submit") == business.count("effect") == 0
            assert "work" in rebuilt.config.tool_policy.disallowed_tools
        else:
            child_state = store.read_state(sid)[0]
            attempt = next(op for op in child_state.operations.values() if op.kind == "tool").attempts[1]
            assert attempt.state == "parked" and attempt.wait["handle"]["provider"] == "work"
            assert business.count("submit") == business.count("effect") == 1
            server.close()
            server = Server(store, business, config=config, agent=parent)
            server.resume()
            rt = server.server.kernel.runtime(parent, server.host.build_run_config(None))
            rebuilt = rt.child_runtime(store, sid)
            assert rebuilt.providers["work"] is rt.providers["work"]
            assert business.count("submit") == business.count("effect") == 1
            server.rpc("turn/interrupt", {"threadId": THREAD, "expectedTurnId": TURN, "actionId": "cancel-parent"})
            server.join()
            drive(store, sid, runtime=rebuilt)
            assert any(
                row.record.kind == "input_applied"
                and row.record.payload["input"]["kind"] == "control"
                and row.record.payload["input"]["payload"]["action"] == "cancel"
                for row in store.read_state(sid)[1]
            )
            assert business.count("cancel") == 1
    finally:
        server.close()


def test_provider_fixture_comes_from_real_app_server():
    canonical = Path(__file__).parents[1] / "fixtures/parity/app_server_providers.json"
    assert produce() == json.loads(canonical.read_text())


@pytest.mark.persistent_store
@pytest.mark.parametrize("cut,supported", [("accepted", True), ("parked", True), ("accepted", False)])
def test_real_sigkill_at_accept_and_parked_boundaries(database, store, business, cut, supported):
    # Install the schema before spawning; the subprocess only opens this store.
    assert store is not None
    worker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.app_server_provider_fixtures",
            "--worker-store",
            str(database),
            "--business",
            str(business.path),
            "--cut",
            cut,
            "--supported",
            str(int(supported)),
        ],
        cwd=Path(__file__).parents[2],
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if cut == "accepted":
            wait_for(lambda: business.setting("accepted") is True)
            _, records, _ = store.read_state(THREAD)
            assert not any(row.record.kind == "op_parked" for row in records)
        else:
            wait_for(lambda: business.setting("parked") is True)
            assert any(row.record.kind == "op_parked" for row in store.read_state(THREAD)[1])
        worker.kill()
        worker.wait(timeout=5)
        assert worker.returncode == -9
        assert business.count("effect") == 1
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
        stdout, stderr = worker.communicate(timeout=5)
        assert not stdout and not stderr, (stdout, stderr)
    business.setting("pause_after_accept", False)
    server = Server(store, business, supported=supported)
    try:
        # Real lease expiry, without editing the store's lease or clock.
        Event().wait(15.1)
        server.resume()
        if cut == "accepted" and not supported:
            assert server.attempt().state == "unknown"
            assert server.attempt().unknown.payload["retry"] == "stop"
            assert business.count("submit") == 1
        else:
            assert server.attempt().state == "parked"
            assert business.count("submit") == (2 if cut == "accepted" else 1)
            business.finish()
            server.due()
            server.resume()
            assert server.attempt().state == "completed"
        assert business.count("effect") == 1 and business.count("sync") == 0
    finally:
        server.close()
