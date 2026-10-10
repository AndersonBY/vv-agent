"""Real-store kernel recovery; external calls are always the scripted client."""

# Fork-free subprocesses communicate exact fault positions through duplex pipes.
import multiprocessing
import os
import signal
import time
from dataclasses import replace
from pathlib import Path

import pytest

from vv_agent.agent import Agent
from vv_agent.config import ResolvedModelConfig
from vv_agent.events import ModelCallCompletedEvent
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.run_config import RunConfig
from vv_agent.session.kernel import Runtime, drive, read_state
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.store import LeaseLost
from vv_agent.session.supervisor import tick
from vv_agent.tools.function import FunctionTool, function_tool
from vv_agent.tools.metadata import ToolIdempotency, ToolMetadata
from vv_agent.tools.outputs import ToolOutputText
from vv_agent.types import LLMResponse, ToolCall

from .conftest import open_store
from .controlled import ControlledProvider

MODEL_USAGE = {"prompt_tokens": 12000, "completion_tokens": 300, "prompt_tokens_details": {"cached_tokens": 0}}


def runtime(database, steps, tools=(), **kwargs):
    return Runtime(
        agent=Agent(name="kernel", instructions="Be precise.", tools=list(tools)),
        config=RunConfig(workspace=Path("/tmp"), session_memory_enabled=False),
        resolved=ResolvedModelConfig("scripted", "scripted", "scripted", "scripted", []),
        llm=ScriptedLLM(list(steps)),
        heartbeat_store=lambda: open_store(database),
        **kwargs,
    )


def start(store, sid="s"):
    with store.atomic() as tx:
        tx.create(SessionSpec(sid, "test", "/tmp"), consumers=("events",))
        tx.push(sid, InboxItem("initial", "user", {"content": "go"}))


def test_completed_results_are_reused(store, database):
    effects = []

    @function_tool
    def effect() -> str:
        effects.append(1)
        return "ok"

    start(store)
    rt = runtime(database, [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")], [effect])
    drive(store, "s", runtime=rt)
    assert effects == [1]
    assert not rt.llm.steps
    drive(store, "s", runtime=runtime(database, []))
    state, records, _ = read_state(store, "s")
    assert state.active_turn_id is None
    assert [r.record.payload["status"] for r in records if r.record.kind == "turn_ended"] == ["completed"]


def test_model_unknown_has_only_one_retry(store, database):
    calls = []

    def lost(request):
        calls.append(request)
        raise TimeoutError("ack lost")

    start(store)
    drive(store, "s", runtime=runtime(database, [lost, lost]))
    state, records, _ = read_state(store, "s")
    assert len(calls) == 2
    assert len(state.operations) == 1
    unknowns = [r.record for r in records if r.record.kind == "op_unknown"]
    assert len(unknowns) == 2
    assert all(r.payload["duplicate_cost_risk"] for r in unknowns)
    assert all(r.payload["observation"]["code"] == "duplicate_model_request_and_cost" for r in unknowns)
    assert state.active_turn_id is None


def force_expire(database):
    with open_store(database) as store, store.atomic():
        store._rows("UPDATE sk_session SET lease_until_ms=0 WHERE lease_owner IS NOT NULL")


def worker(
    database,
    pipe,
    stop_point=None,
    stop_kind=None,
    *,
    mode="definitive",
    idempotent=False,
    long_tool=None,
    two_tools=False,
    stop_tool=None,
    retain_model=False,
):
    provider = ControlledProvider(database, mode, idempotent=idempotent)

    def function(context, args):
        if long_tool:
            stopped = __import__("threading").Event()
            context.ctx.cancellation_token.on_cancel(lambda: (pipe.send(("token", time.monotonic())), stopped.set()))
            pipe.send(("tool_running", None))
            if long_tool == "cooperative":
                assert stopped.wait(10)
                context.ctx.check_cancelled()
            else:
                __import__("threading").Event().wait(10)
        return "ok"

    tool = FunctionTool(
        name="effect",
        description="effect",
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke=function,
        tool_metadata=ToolMetadata(idempotency=ToolIdempotency.SUPPORTED if idempotent else ToolIdempotency.UNSUPPORTED),
    )
    calls = [ToolCall("a", "effect", {})]
    if two_tools:
        calls.append(ToolCall("b", "effect", {}))
    fired = False

    def hook(point, record):
        nonlocal fired
        if (
            not fired
            and point == stop_point
            and (stop_kind is None or record.kind == stop_kind)
            and (stop_tool is None or ("/tool/" in (record.operation_id or "")) == stop_tool)
        ):
            fired = True
            pipe.send(("barrier", record.to_dict() if record else None))
            assert pipe.recv() == "continue"

    def model_response(_request):
        response = LLMResponse("first", raw={"usage": MODEL_USAGE})
        with open_store(database) as reader:
            plan = records_of(reader, "op_planned")[-1]
        provider.retain_model_result(plan, {"content": response.content, "tool_calls": [], "raw": response.raw})
        return response

    rt = runtime(
        database,
        [model_response if retain_model else LLMResponse("", calls), LLMResponse("done")],
        [tool],
        hook=hook,
        providers={} if long_tool else {"effect": provider},
    )
    try:
        with open_store(database) as store:
            drive(store, "s", runtime=rt)
        pipe.send(("done", None))
    except LeaseLost:
        pipe.send(("fenced", None))
    finally:
        pipe.close()


def spawn_worker(database, **kwargs):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=worker, args=(database, child), kwargs=kwargs)
    proc.start()
    child.close()
    return proc, parent


def receive(pipe, expected, timeout=12):
    assert pipe.poll(timeout), f"worker never reached {expected}"
    kind, value = pipe.recv()
    assert kind == expected, (kind, value)
    return value


def kill(proc, pipe, database):
    proc.kill()
    proc.join(5)
    assert not proc.is_alive()
    pipe.close()
    force_expire(database)


def effect_tool(*, idempotent=False, approval=False):
    return FunctionTool(
        name="effect",
        description="effect",
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke=lambda _ctx, _args: ToolOutputText("ok"),
        needs_approval=approval,
        tool_metadata=ToolMetadata(idempotency=ToolIdempotency.SUPPORTED if idempotent else ToolIdempotency.UNSUPPORTED),
    )


def records_of(store, kind):
    return [r.record for r in read_state(store, "s")[1] if r.record.kind == kind]


@pytest.mark.parametrize("cut", ["op_planned", "op_completed"])
@pytest.mark.persistent_store
def test_kill_between_tools_reuses_model_and_completed_tool(store, database, cut):
    provider = ControlledProvider(database)
    provider.install()
    start(store)
    # First cut: model result and both plans committed. Second: A's result committed.
    proc, pipe = spawn_worker(database, stop_point="after_commit", stop_kind=cut, two_tools=True, stop_tool=True)
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        calls_before = provider.counts()[0]
        rt = runtime(database, [LLMResponse("done")], [effect_tool()], providers={"effect": provider})
        drive(store, "s", runtime=rt)
        assert provider.counts() == (2, 2)
        assert calls_before in {0, 1}
        assert len(records_of(store, "op_completed")) == 4
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


@pytest.mark.parametrize("idempotent", [False, True])
def test_unknown_tools_resend_only_declared_idempotency(store, database, idempotent):
    provider = ControlledProvider(database, "unknown", idempotent=idempotent)
    provider.install()
    start(store)
    drive(
        store,
        "s",
        runtime=runtime(
            database,
            [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
            [effect_tool(idempotent=idempotent)],
            providers={"effect": provider},
        ),
    )
    assert provider.counts() == (2 if idempotent else 1, 1)
    with open_store(database) as reader:
        keys = [row[0] for row in reader._rows("SELECT key FROM provider_calls")]
    assert len(set(keys)) == 1
    assert records_of(store, "op_unknown")[0].payload["observation"]["code"] == "tool_outcome_unknown"


@pytest.mark.persistent_store
def test_parked_job_survives_kill_with_same_handle(store, database):
    provider = ControlledProvider(database, "accepted")
    provider.install()
    start(store)
    proc, pipe = spawn_worker(database, stop_point="after_commit", stop_kind="op_parked", mode="accepted")
    try:
        parked = receive(pipe, "barrier")
        kill(proc, pipe, database)
        drive(store, "s", runtime=runtime(database, [], [effect_tool()], providers={"effect": provider}))
        assert not records_of(store, "op_unknown")
        assert records_of(store, "op_parked")[0].payload == parked["payload"]
        with open_store(database) as provider_store, provider_store.atomic():
            provider_store._rows("UPDATE provider_jobs SET ready=true")
        drive(store, "s", runtime=runtime(database, [LLMResponse("done")], [effect_tool()], providers={"effect": provider}))
        assert provider.counts() == (1, 1)
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


@pytest.mark.parametrize("cooperative", [True, False])
@pytest.mark.persistent_store
def test_heartbeat_cancels_blocked_tool_under_two_seconds(store, database, cooperative):
    start(store)
    proc, pipe = spawn_worker(database, long_tool="cooperative" if cooperative else "unstoppable", two_tools=True)
    try:
        receive(pipe, "tool_running")
        before = time.monotonic()
        with store.atomic() as tx:
            tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, target_turn_id="s/turn/initial"))
        observed = receive(pipe, "token", 2)
        assert observed - before <= 2
        receive(pipe, "done")
        proc.join(5)
        assert records_of(store, "turn_ended")[0].payload["status"] == "cancelled"
        starts = records_of(store, "op_started")
        assert len(starts) == 2  # one model, one tool; B and the next model never dispatch
        unknowns = records_of(store, "op_unknown")
        assert bool(unknowns) is not cooperative
        if cooperative:
            assert any(r.payload["evidence"] == ["cooperative-stop"] for r in records_of(store, "op_completed"))
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


@pytest.mark.persistent_store
def test_acceptance_before_park_commit_recovers_from_durable_evidence(store, database):
    provider = ControlledProvider(database, "accepted")
    provider.install()
    start(store)
    proc, pipe = spawn_worker(database, stop_point="after_external_call", stop_tool=True, mode="accepted")
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        with store.atomic() as tx:
            tx.push("s", provider.callback(kind="provider_evidence"))
        drive(store, "s", runtime=runtime(database, [], [effect_tool()], providers={"effect": provider}))
        assert provider.counts() == (1, 1)
        assert not records_of(store, "op_unknown")
        assert len(records_of(store, "op_parked")) == 1
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


@pytest.mark.parametrize("idempotent", [False, True])
@pytest.mark.persistent_store
def test_zombie_after_admission_is_fenced_but_external_call_can_still_happen(store, database, idempotent):
    provider = ControlledProvider(database, idempotent=idempotent)
    provider.install()
    start(store)
    proc, pipe = spawn_worker(database, stop_point="before_external_call", stop_tool=True, idempotent=idempotent)
    try:
        receive(pipe, "barrier")
        # Acquire the row first: SIGSTOP must never freeze a heartbeat holding its lock.
        with store.atomic():
            store._lock("s")
            os.kill(proc.pid, signal.SIGSTOP)
            assert os.WIFSTOPPED(os.waitpid(proc.pid, os.WUNTRACED)[1])
            store.connection.execute("UPDATE sk_session SET lease_until_ms=0 WHERE session_id='s'")
        drive(
            store,
            "s",
            runtime=runtime(
                database, [LLMResponse("done")], [effect_tool(idempotent=idempotent)], providers={"effect": provider}
            ),
        )
        before = len(read_state(store, "s")[1])
        assert provider.counts()[0] == int(idempotent)
        pipe.send("continue")
        os.kill(proc.pid, signal.SIGCONT)
        receive(pipe, "fenced")
        proc.join(5)
        assert len(read_state(store, "s")[1]) == before
        assert provider.counts() == (2 if idempotent else 1, 1)
        assert len(records_of(store, "op_unknown")) == 1
    finally:
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGCONT)
            kill(proc, pipe, database)


@pytest.mark.parametrize("when", ["before_unknown", "after_unknown", "after_consumed", "after_terminal"])
@pytest.mark.persistent_store
def test_authenticated_late_results_have_one_tool_message_and_correction_only_when_consumed(store, database, when):
    provider = ControlledProvider(database)
    provider.install()
    start(store)
    proc, pipe = spawn_worker(database, stop_point="after_external_call", stop_tool=True)
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        pushed = False
        requests = []

        def push():
            nonlocal pushed
            if not pushed:
                with store.atomic() as tx:
                    tx.push("s", provider.callback())
                pushed = True

        def hook(point, record):
            if when == "after_unknown" and point == "after_commit" and record.kind == "op_unknown":
                push()
            if (
                when == "after_consumed"
                and point == "after_commit"
                and record.kind == "op_planned"
                and record.payload["op_kind"] == "model"
            ):
                push()

        def answer(request):
            requests.append(request)
            return LLMResponse("done")

        if when == "before_unknown":
            push()
        rt = runtime(database, [answer, answer], [effect_tool()], providers={"effect": provider}, hook=hook)
        drive(store, "s", runtime=rt)
        if when == "after_terminal":
            push()
            drive(store, "s", runtime=runtime(database, [], [effect_tool()], providers={"effect": provider}))
        results = [r for r in records_of(store, "op_completed") if "/tool/" in r.operation_id]
        assert len(results) == 1
        context = {
            "before_unknown": "normal",
            "after_unknown": "normal",
            "after_consumed": "correction",
            "after_terminal": "audit",
        }[when]
        assert results[0].payload["context"] == context
        assert provider.counts() == (1, 1)
        for request in requests:
            tools = [m for m in request.messages if m.role == "tool"]
            assert len(tools) == 1
        if when == "after_consumed":
            assert len(requests) == 2
            assert "tool_outcome_unknown" in next(m.content for m in requests[0].messages if m.role == "tool")
            assert sum(m.content.startswith("Correction for") for m in requests[1].messages) == 1
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


@pytest.mark.persistent_store
def test_steer_during_batch_follow_up_and_final_watermark(store, database):
    provider = ControlledProvider(database)
    provider.install()
    start(store)
    proc, pipe = spawn_worker(database, stop_point="after_commit", stop_kind="op_completed", stop_tool=True, two_tools=True)
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        with store.atomic() as tx:
            tx.push("s", InboxItem("steer", "steer", {"content": "steer now"}, target_turn_id="s/turn/initial"))
            tx.push("s", InboxItem("follow", "follow_up", {"content": "next task"}))
        requests = []
        inserted = False

        def hook(point, record):
            nonlocal inserted
            if point == "before_turn_end" and not inserted:
                inserted = True
                with store.atomic() as tx:
                    tx.push("s", InboxItem("last", "steer", {"content": "at final boundary"}, target_turn_id=record.turn_id))

        def answer(request):
            requests.append(request)
            return LLMResponse("done")

        drive(
            store,
            "s",
            runtime=runtime(database, [answer, answer, answer], [effect_tool()], providers={"effect": provider}, hook=hook),
        )
        assert provider.counts() == (2, 2)
        assert len(requests) == 3
        texts = [m.content for m in requests[0].messages]
        assert texts[-1] == "steer now"
        assert [m.role for m in requests[0].messages][-3:] == ["tool", "tool", "user"]
        assert requests[1].messages[-1].content == "at final boundary"
        assert requests[2].messages[-1].content == "next task"
        assert len(records_of(store, "turn_started")) == 2
        assert not store.peek_inbox("s")
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


def test_approval_park_restart_deny_and_exact_reply(store, database):
    provider = ControlledProvider(database)
    provider.install()
    start(store)
    tool = effect_tool(approval=lambda _context, _args: True)
    drive(
        store,
        "s",
        runtime=runtime(database, [LLMResponse("", [ToolCall("a", "effect", {})])], [tool], providers={"effect": provider}),
    )
    assert provider.counts() == (0, 0)
    park = records_of(store, "op_parked")[0]
    h = park.payload["handle"]
    answer = InboxItem(
        "approval",
        "approval_answer",
        {
            "operation_id": park.operation_id,
            "attempt": park.attempt,
            "request_id": h["request_id"],
            "request_digest": h["request_digest"],
            "scope": h["scope"],
            "decision": "approve",
        },
        target_turn_id=park.turn_id,
    )
    with store.atomic() as tx:
        tx.push("s", replace(answer, input_id="wrong", payload=answer.payload | {"scope": ["other"]}))
        tx.push("s", answer)
        tx.push("s", answer)
    drive(store, "s", runtime=runtime(database, [LLMResponse("done")], [tool], providers={"effect": provider}))
    assert provider.counts() == (1, 1)
    assert records_of(store, "input_applied")[1].payload["disposition"] == "rejected"
    assert_event_projection(store)


def test_ask_user_is_same_turn_parked_interaction(store, database):
    start(store)
    drive(store, "s", runtime=runtime(database, [LLMResponse("", [ToolCall("q", "ask_user", {"question": "Which?"})])]))
    park = records_of(store, "op_parked")[0]
    assert park.payload["phase"] == "before_dispatch"
    assert not records_of(store, "turn_ended")
    with store.atomic() as tx:
        tx.push(
            "s",
            InboxItem(
                "reply",
                "user",
                {
                    "content": {
                        "operation_id": park.operation_id,
                        "interaction_id": park.payload["handle"]["interaction_id"],
                        "text": "blue",
                    }
                },
                target_turn_id=park.turn_id,
            ),
        )
    seen = []

    def answer(request):
        seen.extend(request.messages)
        return LLMResponse("done")

    drive(store, "s", runtime=runtime(database, [answer]))
    assert len(records_of(store, "turn_started")) == 1
    assert len([m for m in seen if m.role == "tool" and m.content == "blue"]) == 1
    assert not [m for m in seen if m.role == "user" and m.content == "blue"]
    assert_event_projection(store)


@pytest.mark.persistent_store
def test_empty_inbox_started_model_recovered_by_paginated_supervisor(store, database):
    start(store)
    proc, pipe = spawn_worker(database, stop_point="before_external_call", stop_tool=False)
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        assert not store.peek_inbox("s")
        projected = []

        def project(sid, consumer):
            with store.atomic() as tx:
                batch = tx.consumer_batch(sid, consumer)
                projected.extend(batch.records)
                tx.ack(batch)

        tick(store, runtime=lambda _sid: runtime(database, [LLMResponse("done")], [effect_tool()]), project=project, page_size=1)
        assert len(records_of(store, "op_started")) == 2
        assert records_of(store, "op_started")[1].attempt == 2
        assert any(r.record.kind == "turn_ended" for r in projected)
        assert not store.list_runnable()
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


def test_cancel_parked_provider_keeps_unknown_and_audit_late_result(store, database):
    provider = ControlledProvider(database, "accepted")
    provider.install()
    start(store)
    drive(
        store,
        "s",
        runtime=runtime(
            database, [LLMResponse("", [ToolCall("a", "effect", {})])], [effect_tool()], providers={"effect": provider}
        ),
    )
    with store.atomic() as tx:
        tx.push("s", InboxItem("cancel", "control", {"action": "cancel"}, target_turn_id="s/turn/initial"))
    drive(store, "s", runtime=runtime(database, [], [effect_tool()], providers={"effect": provider}))
    assert len(records_of(store, "op_unknown")) == 1
    assert records_of(store, "turn_ended")[0].payload["status"] == "cancelled"
    with store.atomic() as tx:
        tx.push("s", provider.callback())
    drive(store, "s", runtime=runtime(database, [], [effect_tool()], providers={"effect": provider}))
    assert records_of(store, "op_completed")[-1].payload["context"] == "audit"


def test_real_function_timeout_is_unknown_not_definitive_error(store, database):
    import threading

    stopped = threading.Event()
    entered = threading.Event()

    @function_tool(timeout_seconds=0.05)
    def effect() -> str:
        entered.set()
        assert stopped.wait(10)
        return "late effect"

    start(store)
    try:
        drive(
            store, "s", runtime=runtime(database, [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")], [effect])
        )
        assert entered.is_set()
        assert len(records_of(store, "op_unknown")) == 1
        assert not [r for r in records_of(store, "op_completed") if "/tool/" in r.operation_id]
    finally:
        stopped.set()


@pytest.mark.parametrize("mode", ["deny", "revoke", "invalid_args", "version"])
def test_tool_admission_reuses_approval_policy_validation_and_version(store, database, mode):
    from vv_agent.run_config import ToolPolicy

    provider = ControlledProvider(database)
    provider.install()
    start(store)
    tool = effect_tool(approval=mode in {"deny", "revoke", "version"})
    args = {"surprise": True} if mode == "invalid_args" else {}
    drive(
        store,
        "s",
        runtime=runtime(
            database,
            [LLMResponse("", [ToolCall("a", "effect", args)]), LLMResponse("done")],
            [tool],
            providers={"effect": provider},
        ),
    )
    if mode != "invalid_args":
        parked = records_of(store, "op_parked")[0]
        handle = parked.payload["handle"]
        with store.atomic() as tx:
            tx.push(
                "s",
                InboxItem(
                    "answer",
                    "approval_answer",
                    {
                        "operation_id": parked.operation_id,
                        "attempt": parked.attempt,
                        "request_id": handle["request_id"],
                        "request_digest": handle["request_digest"],
                        "scope": handle["scope"],
                        "decision": "deny" if mode == "deny" else "approve",
                    },
                    target_turn_id=parked.turn_id,
                ),
            )
        rt = runtime(database, [LLMResponse("done")], [tool], providers={"effect": provider})
        if mode == "revoke":
            rt.config.tool_policy = ToolPolicy(can_use_tool=lambda _name, _args: False)
        if mode == "version":
            rt.handler_version = "2"
        drive(store, "s", runtime=rt)
    assert provider.counts() == (0, 0)
    assert records_of(store, "turn_ended")
    if mode == "version":
        assert records_of(store, "turn_ended")[0].payload["reason"] == "handler_version_mismatch"


def test_budget_unknown_measurement_prevents_retry_under_strict_limits(store, database):
    from vv_agent.budget import RunBudgetLimits, UnavailableMetricPolicy

    start(store)
    calls = []

    def lost(request):
        calls.append(request)
        raise TimeoutError

    rt = runtime(database, [lost, lost])
    rt.config.budget_limits = RunBudgetLimits(max_total_tokens=1000, unavailable_metric_policy=UnavailableMetricPolicy.STOP)
    drive(store, "s", runtime=rt)
    assert len(calls) == 1
    end = records_of(store, "turn_ended")[0]
    assert end.payload["reason"] == "budget_exhausted"
    assert end.payload["budget"]["total_tokens"] is None


def test_run_events_are_stable_typed_projections_and_do_not_execute(store, database):
    from vv_agent.events import event_from_dict
    from vv_agent.session.projection import project_records

    start(store)
    drive(store, "s", runtime=runtime(database, [LLMResponse("done")]))
    _, rows, _ = read_state(store, "s")
    first = [event.to_dict() for event in project_records(rows)]
    second = [event.to_dict() for event in project_records(rows)]
    assert first == second
    assert [e["type"] for e in first] == [
        "run_started",
        "agent_started",
        "cycle_started",
        "model_call_started",
        "diagnostic",
        "model_call_completed",
        "diagnostic",
        "run_completed",
    ]
    assert len({e["event_id"] for e in first}) == len(first)
    assert all(event_from_dict(e).to_dict() == e for e in first)


def test_client_retries_are_owned_by_kernel(store, database):
    from vv_agent.model_settings import ModelSettings, RetrySettings

    start(store)
    requests = []

    def answer(request):
        requests.append(request)
        return LLMResponse("done")

    rt = runtime(database, [answer])
    rt.config.model_settings = ModelSettings(retry=RetrySettings(max_attempts=9))
    drive(store, "s", runtime=rt)
    assert requests[0].model_settings.retry.max_attempts == 1


def test_lost_commit_ack_reuses_real_retained_model_result(store, database, monkeypatch):
    from contextlib import contextmanager

    class AckLost(BaseException):
        pass

    atomic = store.atomic

    @contextmanager
    def lost_ack():
        with atomic() as tx:
            yield tx
        if records_of(store, "op_completed"):
            raise AckLost

    start(store)
    with monkeypatch.context() as patch:
        patch.setattr(store, "atomic", lost_ack)
        with pytest.raises(AckLost):
            drive(store, "s", runtime=runtime(database, [LLMResponse("retained")]))
    drive(store, "s", runtime=runtime(database, []))
    assert len(records_of(store, "op_started")) == 1
    assert records_of(store, "turn_ended")[0].payload["result"] == "retained"


def test_twenty_concurrent_callback_replays_and_different_bytes_conflict(store, database):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from vv_agent.session.store import Conflict

    provider = ControlledProvider(database, "accepted")
    provider.install()
    start(store)
    drive(
        store,
        "s",
        runtime=runtime(
            database, [LLMResponse("", [ToolCall("a", "effect", {})])], [effect_tool()], providers={"effect": provider}
        ),
    )
    item = provider.callback()
    barrier = Barrier(20)

    def deliver(_index):
        with open_store(database) as other:
            barrier.wait(timeout=10)
            with other.atomic() as tx:
                return tx.push("s", item)

    with ThreadPoolExecutor(max_workers=20) as pool:
        receipts = list(pool.map(deliver, range(20)))
    assert sum(not r.replayed for r in receipts) == 1
    assert len({r.input_seq for r in receipts}) == 1
    with pytest.raises(Conflict), store.atomic() as tx:
        tx.push("s", replace(item, payload=item.payload | {"result": {"different": True}}))
    drive(store, "s", runtime=runtime(database, [LLMResponse("done")], [effect_tool()], providers={"effect": provider}))
    assert provider.counts() == (1, 1)
    assert len([r for r in records_of(store, "input_applied") if r.payload["input"]["input_id"] == item.input_id]) == 1


@pytest.mark.persistent_store
def test_future_retry_not_runnable_until_database_due_time(store, database):
    from .helpers import record

    start(store)
    proc, pipe = spawn_worker(database, stop_point="before_external_call", stop_tool=False)
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        state, rows, watermark = read_state(store, "s")
        plan = next(iter(state.operations.values())).attempts[1].plan
        assert plan.turn_id is not None and plan.operation_id is not None
        lease = store.acquire("s", owner="test-repair", ttl_ms=15000)
        due = store.renew(lease, ttl_ms=15000).db_now_ms + 1000
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=rows[-1].seq,
                commit_id="future-repair",
                expected_inbox_seq=watermark,
                records=(record("op_unknown", tid=plan.turn_id, oid=plan.operation_id, retry_at_ms=due),),
            )
        store.release(lease)
        assert not [w for w in store.list_runnable() if w.kind == "drive"]
        rt = runtime(database, [LLMResponse("done")], [effect_tool()])
        drive(store, "s", runtime=rt)
        assert len(rt.llm.steps) == 1
        # Poll the actual database clock; this is a due-time test, not a guessed kill window.
        deadline = time.monotonic() + 5
        while store._now() < due:
            assert time.monotonic() < deadline
            __import__("threading").Event().wait(0.01)
        assert [w for w in store.list_runnable() if w.kind == "drive"]
        drive(store, "s", runtime=rt)
        assert not rt.llm.steps
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


def test_budget_counts_and_elapsed_survive_recovery(store, database):
    from vv_agent.budget import RunBudgetLimits

    provider = ControlledProvider(database)
    provider.install()
    start(store)
    cut = False

    class Restart(BaseException):
        pass

    def hook(point, r):
        nonlocal cut
        if not cut and point == "after_commit" and r.kind == "op_completed" and "/tool/" in (r.operation_id or ""):
            cut = True
            raise Restart

    rt = runtime(
        database,
        [LLMResponse("", [ToolCall("a", "effect", {})])],
        [effect_tool()],
        providers={"effect": provider},
        hook=hook,
    )
    rt.config.budget_limits = RunBudgetLimits(max_tool_calls=1)
    with pytest.raises(Restart):
        drive(store, "s", runtime=rt)
    observed_before = sum(r.payload["usage"]["elapsed_ms"] for r in read_state(store, "s")[0].usage_values.values())
    drive(
        store,
        "s",
        runtime=runtime(
            database, [LLMResponse("", [ToolCall("b", "effect", {})])], [effect_tool()], providers={"effect": provider}
        ),
    )
    end = records_of(store, "turn_ended")[0]
    assert end.payload["budget"]["tool_calls"] == 1
    assert end.payload["budget"]["elapsed_ms"] >= observed_before > 0
    assert provider.counts() == (1, 1)
    assert end.payload["reason"] == "budget_exhausted"


@pytest.mark.parametrize("retry_started", [False, True])
@pytest.mark.persistent_store
def test_late_first_model_result_respects_retry_dispatch_boundary(store, database, retry_started):
    from vv_agent.session.projection import project_records
    from vv_agent.session.result import project_result
    from vv_agent.session.runtime import model_usage

    provider = ControlledProvider(database)
    provider.install()
    start(store)
    proc, pipe = spawn_worker(database, stop_point="after_external_call", stop_tool=False, retain_model=True)
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        pushed = False

        def hook(point, record):
            nonlocal pushed
            desired = "before_external_call" if retry_started else "after_commit"
            if not pushed and point == desired and record.attempt == 2:
                pushed = True
                with store.atomic() as tx:
                    tx.push("s", provider.callback())

        rt = runtime(database, [LLMResponse("second")], [effect_tool()], providers={"model": provider}, hook=hook)
        drive(store, "s", runtime=rt)
        assert len(rt.llm.steps) == (0 if retry_started else 1)
        result = next(r for r in records_of(store, "op_completed") if r.attempt == 1)
        assert result.payload["context"] == ("audit" if retry_started else "normal")
        assert result.payload["usage"] == MODEL_USAGE
        assert records_of(store, "turn_ended")[0].payload["result"] == ("second" if retry_started else "first")
        assert len(records_of(store, "op_started")) == (2 if retry_started else 1)
        state, rows, _ = read_state(store, "s")
        projected = project_result(store, "s", result.turn_id)
        call = next(c for c in projected.token_usage.model_calls if c.attempt == 1)
        assert call.usage == model_usage(MODEL_USAGE)
        assert call.usage.input_tokens == 12000 and call.usage.output_tokens == 300
        assert call.usage.usage_source.value == "provider_reported"
        completed = [e for e in project_records(rows) if isinstance(e, ModelCallCompletedEvent) and e.attempt == 1]
        assert len(completed) == 1 and completed[0].usage == call.usage
        assert len({c.call_id for c in projected.token_usage.model_calls}) == (2 if retry_started else 1)

        # Redelivery under either input identity must not create a second billable call.
        with store.atomic() as tx:
            first_receipt = tx.push("s", provider.callback())
            assert tx.push("s", provider.callback()) == first_receipt
            tx.push("s", provider.callback(input_id="redelivery"))
        drive(store, "s", runtime=rt)
        state, replay_rows, _ = read_state(store, "s")
        assert state.applied_inputs["redelivery"].payload["disposition"] == "noop"
        assert project_result(store, "s", result.turn_id).token_usage == projected.token_usage
        assert len([e for e in project_records(replay_rows) if isinstance(e, ModelCallCompletedEvent) and e.attempt == 1]) == 1
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


@pytest.mark.persistent_store
def test_repair_missing_plans_uses_retained_response_without_model_replay(store, database):
    from vv_agent.session.records import digest

    from .helpers import record

    provider = ControlledProvider(database)
    provider.install()
    start(store)
    proc, pipe = spawn_worker(database, stop_point="before_external_call", stop_tool=False)
    try:
        receive(pipe, "barrier")
        kill(proc, pipe, database)
        state, rows, _ = read_state(store, "s")
        plan = next(iter(state.operations.values())).attempts[1].plan
        assert plan.turn_id is not None and plan.operation_id is not None
        lease = store.acquire("s", owner="retained-result", ttl_ms=15000)
        result = {"content": "", "tool_calls": [ToolCall("a", "effect", {}).to_dict()], "raw": {}}
        with store.atomic() as tx:
            tx.append(
                "s",
                lease=lease,
                expected_seq=rows[-1].seq,
                commit_id="model-result-only",
                records=(
                    record(
                        "op_completed",
                        tid=plan.turn_id,
                        oid=plan.operation_id,
                        result=result,
                        result_digest=digest(result),
                        request_digest=plan.payload["request_digest"],
                        provider_binding="model",
                    ),
                ),
            )
        store.release(lease)
        drive(store, "s", runtime=runtime(database, [LLMResponse("done")], [effect_tool()], providers={"effect": provider}))
        assert provider.counts() == (1, 1)
        assert len([r for r in records_of(store, "op_planned") if r.payload["op_kind"] == "model"]) == 2
    finally:
        if proc.is_alive():
            kill(proc, pipe, database)


def test_abort_before_heartbeat_retains_durable_action(store, database):
    start(store)
    fired = False

    def hook(point, record):
        nonlocal fired
        if not fired and point == "after_commit" and record.kind == "turn_started":
            fired = True
            with store.atomic() as tx:
                tx.push("s", InboxItem("abort", "control", {"action": "abort"}, target_turn_id=record.turn_id))

    drive(store, "s", runtime=runtime(database, [], hook=hook))
    assert records_of(store, "turn_ended")[0].payload["status"] == "aborted"


def test_projection_identity_is_scoped_to_session(store, database):
    from vv_agent.session.projection import project_records
    from vv_agent.session.records import digest, make_record
    from vv_agent.session.store import StoredRecord

    # record_id is only session-scoped; callers may use the same turn id in another session.
    record = make_record(
        "turn_started",
        session_id="s",
        turn_id="same-turn",
        payload={
            "input_ids": [],
            "definition": {"task": {"user_prompt": "x"}},
            "definition_digest": digest({"task": {"user_prompt": "x"}}),
            "handler_version": "1",
            "budget": {},
            "binding": None,
            "generation": 0,
        },
    )
    a = project_records([StoredRecord(record, 1, "c", 1, 1)])[0]
    b = project_records([StoredRecord(replace(record, session_id="other"), 1, "c", 1, 1)])[0]
    assert a.event_id != b.event_id


def test_missing_current_cost_observation_never_reuses_old_meter_value(store, database):
    from vv_agent.budget import HostCost, RunBudgetLimits, UnavailableMetricPolicy

    class Meter:
        available = True

        def read(self):
            return HostCost("credits", 1) if self.available else None

    meter = Meter()
    start(store)

    def hook(point, record):
        if point == "after_commit" and record.kind == "op_planned" and record.payload["op_kind"] == "tool":
            meter.available = False

    rt = runtime(database, [LLMResponse("", [ToolCall("a", "effect", {})])], [effect_tool()], hook=hook)
    rt.config.host_cost_meter = meter
    rt.config.budget_limits = RunBudgetLimits(
        max_host_cost=HostCost("credits", 100), unavailable_metric_policy=UnavailableMetricPolicy.STOP
    )
    drive(store, "s", runtime=rt)
    assert len(records_of(store, "op_started")) == 1
    assert records_of(store, "turn_ended")[0].payload["reason"] == "budget_exhausted"


def test_steer_during_model_retry_waits_for_next_unfrozen_request(store, database):
    start(store)
    requests = []
    inserted = False

    def lost(request):
        requests.append(request)
        raise TimeoutError

    def answer(request):
        requests.append(request)
        return LLMResponse("done")

    def hook(point, r):
        nonlocal inserted
        if point == "after_commit" and r.kind == "op_unknown" and not inserted:
            inserted = True
            with store.atomic() as tx:
                tx.push("s", InboxItem("during-retry", "steer", {"content": "new instruction"}, target_turn_id=r.turn_id))

    drive(store, "s", runtime=runtime(database, [lost, answer, answer], hook=hook))
    assert len(requests) == 3
    assert requests[0].messages == requests[1].messages
    assert requests[2].messages[-1].content == "new instruction" or any(
        m.role == "user" and m.content == "new instruction" for m in requests[2].messages
    )


def test_strict_budget_keeps_historical_tool_counts_when_meter_disappears(store, database):
    from vv_agent.budget import HostCost, RunBudgetLimits, UnavailableMetricPolicy

    class Meter:
        available = True

        def read(self):
            return HostCost("credits", 1) if self.available else None

    meter = Meter()
    start(store)

    def hook(point, r):
        if point == "after_commit" and r.kind == "op_completed" and "/tool/" in (r.operation_id or ""):
            meter.available = False

    rt = runtime(database, [LLMResponse("", [ToolCall("a", "effect", {})])], [effect_tool()], hook=hook)
    rt.config.host_cost_meter = meter
    rt.config.budget_limits = RunBudgetLimits(
        max_host_cost=HostCost("credits", 100), unavailable_metric_policy=UnavailableMetricPolicy.STOP
    )
    drive(store, "s", runtime=rt)
    assert records_of(store, "turn_ended")[0].payload["budget"]["tool_calls"] == 1


def test_handler_managed_dispatch_cannot_run_during_preflight(store, database):
    effects = []
    tool = effect_tool()
    tool.metadata["policy_managed_by_handler"] = True
    tool.on_invoke = lambda _ctx, _args: (effects.append(1), ToolOutputText("ok"))[1]
    start(store)
    drive(store, "s", runtime=runtime(database, [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")], [tool]))
    assert effects == []
    assert len(records_of(store, "op_started")) == 2  # model requests only
    result = next(r for r in records_of(store, "op_completed") if "/tool/" in r.operation_id)
    assert result.payload["result"]["error_code"] == "session_dispatch_boundary_required"


def test_real_context_and_event_projection_keep_logical_cycle_numbers(store, database):
    from vv_agent.session.projection import project_records
    from vv_agent.tools.base import ToolContext

    cycles = []

    @function_tool
    def effect(context: ToolContext) -> str:
        cycles.append(context.cycle_index)
        return "ok"

    start(store)
    steps = [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("", [ToolCall("b", "effect", {})]), LLMResponse("done")]
    drive(store, "s", runtime=runtime(database, steps, [effect]))
    events = project_records(read_state(store, "s")[1])
    from vv_agent.events import event_from_dict

    assert all(event_from_dict(e.to_dict()).to_dict() == e.to_dict() for e in events)
    assert cycles == [1, 2]
    assert [e.cycle_index for e in events if e.type == "model_call_started"] == [1, 2, 3]


def assert_event_projection(store):
    from vv_agent.events import event_from_dict
    from vv_agent.session.projection import project_records

    events = project_records(read_state(store, "s")[1])
    assert all(event_from_dict(e.to_dict()).to_dict() == e.to_dict() for e in events)


def test_undispatched_model_closure_projects_failure(store, database):
    from vv_agent.session.projection import project_records

    start(store)
    rt = runtime(database, [])

    def hook(point, record):
        if point == "after_commit" and record.kind == "op_planned":
            rt.handler_version = "changed"

    rt.hook = hook
    drive(store, "s", runtime=rt)
    events = project_records(read_state(store, "s")[1])
    assert not [e for e in events if e.type in {"model_call_started", "model_call_completed"}]
    assert len([e for e in events if e.type == "model_call_failed"]) == 1
