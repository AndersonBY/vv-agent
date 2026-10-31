"""Real kernel summary requests, deterministic transport, and all session stores."""

from dataclasses import replace

import pytest

from vv_agent import SummaryChildTool
from vv_agent.session.children import ChildSession
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import SessionSpec, digest
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall, ToolDirective, ToolExecutionResult

from .test_compaction import SUMMARY, history
from .test_recovery_matrix import MODEL_USAGE, records_of, runtime, start


@function_tool
def compact() -> ToolExecutionResult:
    return ToolExecutionResult("", "Compaction requested.", metadata={"request_compaction": True})


@function_tool
def child_task(include_main_summary: bool = True) -> str:
    raise AssertionError("admission must use the kernel")


@function_tool
def finish() -> ToolExecutionResult:
    return ToolExecutionResult("", "done", directive=ToolDirective.FINISH)


@function_tool
def wait() -> ToolExecutionResult:
    return ToolExecutionResult("", "question", directive=ToolDirective.WAIT_USER)


def configured(database, steps, *, child=False, hook=lambda *_: None, initial=True):
    bindings = {}
    if child:
        bindings["child_task"] = SummaryChildTool(
            lambda plan: plan.payload["request"]["arguments"].get("include_main_summary", True),
            lambda plan, summary: ChildSession(SessionSpec("child", "test", "/tmp"), f"child input\n{summary}", background=True),
        )
    rt = runtime(database, steps, [compact, child_task, finish, wait], children=bindings, hook=hook)
    rt.config = replace(rt.config, initial_messages=history() if initial else [])
    rt.memory_manager.compact_threshold = 999999
    rt.memory_manager.keep_recent_messages = 1
    return rt


def call(name="compact", args=None):
    return LLMResponse("", [ToolCall("request", name, args or {})])


def summary_plans(store):
    return [r for r in records_of(store, "op_planned") if r.payload["purpose"] == "compaction"]


def assert_summary_billed_once(store):
    from vv_agent.events import ModelCallCompletedEvent
    from vv_agent.session.projection import project_records
    from vv_agent.session.result import project_result
    from vv_agent.types import ModelCallOperation

    events = [
        event
        for event in project_records(read_state(store, "s")[1])
        if isinstance(event, ModelCallCompletedEvent) and event.operation == ModelCallOperation.MEMORY_COMPACTION
    ]
    assert len(events) == 1 and events[0].usage.input_tokens == 12000
    result = project_result(store, "s", "s/turn/initial")
    charges = [call for call in result.token_usage.model_calls if call.operation == ModelCallOperation.MEMORY_COMPACTION]
    assert len(charges) == 1 and charges[0].usage.input_tokens == 12000


def test_manual_compaction_receipt_and_next_primary(store, database):
    seen = []

    def summary(request):
        assert not request.tools
        return LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})

    def answer(request):
        seen.extend(request.messages)
        assert any(m.name == "memory_summary" for m in request.messages)
        assert any(m.tool_call_id == "request" for m in request.messages)
        return LLMResponse("done")

    start(store)
    drive(store, "s", runtime=configured(database, [call(), summary, answer]))
    assert len(summary_plans(store)) == len(records_of(store, "context_compacted")) == 1
    receipt = next(r for r in records_of(store, "op_completed") if r.payload["result"].get("tool_call_id") == "request")
    assert receipt.payload["result"]["metadata"]["compaction_source"]["operation_id"] == receipt.operation_id
    assert len(receipt.payload["result"]["metadata"]["compaction_source"]["context_digest"]) == 64
    completed = [r for r in records_of(store, "boundary_recorded") if r.payload["stage"] == "memory_completed"]
    assert completed[-1].payload["data"]["event"]["metadata"]["requested_by"] == [receipt.operation_id]
    assert completed[-1].payload["data"]["event"]["changed"]
    assert_summary_billed_once(store)
    assert len([r for r in records_of(store, "op_completed") if r.payload["usage"].get("prompt_tokens") == 12000]) == 1


@pytest.mark.parametrize("case", ["empty", "rejected", "failure"])
def test_manual_noop_or_failure_is_consumed_once(store, database, case):
    def unavailable(_):
        raise TimeoutError("provider unavailable")

    steps = [call()]
    if case == "rejected":
        steps.append(LLMResponse("invalid"))
    if case == "failure":
        steps.extend([unavailable, unavailable])
    steps.append(LLMResponse("done"))
    start(store)
    drive(store, "s", runtime=configured(database, steps, initial=case != "empty"))
    drive(store, "s", runtime=configured(database, [], initial=case != "empty"))
    assert not records_of(store, "context_compacted")
    assert len(summary_plans(store)) == (0 if case == "empty" else 2 if case == "failure" else 1)
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"
    assert any(r.payload["stage"] == "memory_completed" for r in records_of(store, "boundary_recorded"))


@pytest.mark.parametrize("terminal", ["finish", "wait"])
def test_terminal_batch_does_not_request_summary(store, database, terminal):
    start(store)
    drive(
        store,
        "s",
        runtime=configured(database, [LLMResponse("", [ToolCall("request", "compact", {}), ToolCall("end", terminal, {})])]),
    )
    assert not summary_plans(store)
    assert not records_of(store, "context_compacted")
    state = read_state(store, "s")[0]
    assert state.active_turn_id is None if terminal == "finish" else state.phase == "parked"
    if terminal == "wait":
        from vv_agent.session.records import InboxItem

        tid = state.active_turn_id
        assert tid is not None
        parked = state.turns[tid].wait
        assert parked is not None
        with store.atomic() as tx:
            tx.push(
                "s",
                InboxItem(
                    "reply", "user", {"content": {"interaction_id": parked.payload["interaction_id"], "text": "continue"}}, tid, 0
                ),
            )
        drive(store, "s", runtime=configured(database, [LLMResponse(SUMMARY), LLMResponse("done")]))
        assert len(summary_plans(store)) == 1
        assert len(records_of(store, "context_compacted")) == 1


def test_two_requests_and_threshold_share_operation(store, database):
    def summary(request):
        return LLMResponse(SUMMARY)

    def requested(request):
        # Threshold rises only after the first primary plan; frozen settings stay stable.
        return LLMResponse("", [ToolCall("a", "compact", {}), ToolCall("b", "compact", {})])

    start(store)
    rt = configured(database, [requested, summary, LLMResponse("done")])
    # Initial history is below threshold; this long output makes it exceed the threshold.
    rt.memory_manager.compact_threshold = 12000
    rt.llm.steps[0] = LLMResponse("facts " * 14000, [ToolCall("a", "compact", {}), ToolCall("b", "compact", {})])
    drive(store, "s", runtime=rt)
    assert len(summary_plans(store)) == 1
    events = [
        r.payload["data"]["event"] for r in records_of(store, "boundary_recorded") if r.payload["stage"] == "memory_completed"
    ]
    assert len(events[-1]["metadata"]["requested_by"]) == 2


def test_child_fresh_summary_is_billed_and_parent_is_unchanged(store, database):
    seen = []

    def summary(request):
        assert request.metadata["mode"] == "child_summary"
        assert "old facts" in request.messages[0].content
        assert "go" in request.messages[0].content
        return LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})

    def answer(request):
        seen.extend(request.messages)
        return LLMResponse("done")

    start(store)
    drive(store, "s", runtime=configured(database, [call("child_task"), summary, answer], child=True))
    assert not records_of(store, "context_compacted")
    assert any("old facts" in m.content for m in seen)
    assert not any(m.name == "memory_summary" for m in seen)
    assert len(summary_plans(store)) == 1
    child_input = store.peek_inbox("child")[0].item.payload["content"]
    assert "Compressed Agent Memory" in child_input and "Continue the task" in child_input
    assert_summary_billed_once(store)
    assert len([r for r in records_of(store, "op_completed") if r.payload["usage"].get("prompt_tokens") == 12000]) == 1


@pytest.mark.parametrize("flag", [False, True])
def test_child_summary_rejection_or_opt_out(store, database, flag):
    start(store)
    steps = [call("child_task", {"include_main_summary": flag})]
    if flag:
        steps.append(LLMResponse("invalid"))
    steps.append(LLMResponse("done"))
    drive(store, "s", runtime=configured(database, steps, child=True))
    if flag:
        assert not records_of(store, "op_parked")
        assert any(
            r.payload["result"].get("error_code") == "child_summary_unavailable" for r in records_of(store, "op_completed")
        )
    else:
        assert not summary_plans(store)
        assert store.peek_inbox("child")[0].item.payload["content"] == "child input\nNone"


def test_child_summary_unknown_uses_existing_retries_and_never_admits(store, database):
    calls = []

    def unknown(request):
        calls.append((request.metadata["operation_id"], request.metadata["attempt"]))
        raise TimeoutError("unknown")

    start(store)
    drive(store, "s", runtime=configured(database, [call("child_task"), unknown, unknown, LLMResponse("done")], child=True))
    assert len(calls) == 2 and calls[0][0] == calls[1][0] and [v[1] for v in calls] == [1, 2]
    assert len(records_of(store, "op_unknown")) == 2
    assert all(r.payload["duplicate_cost_risk"] and r.payload["measurement_missing"] for r in records_of(store, "op_unknown"))
    assert not records_of(store, "op_parked")


class Crash(BaseException):
    pass


@pytest.mark.parametrize(
    ("child", "point"),
    [(False, p) for p in ("intent", "memory_start", "plan", "start", "receipt", "replacement", "telemetry")]
    + [(True, p) for p in ("plan", "start", "receipt", "admission")],
)
def test_crash_commit_cuts_reuse_summary_and_pairing(store, database, child, point):
    crashed = []
    calls = []

    def summary(request):
        calls.append(request.metadata["attempt"])
        return LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})

    def hook(at, record):
        if at != "after_commit":
            return
        is_summary = record.operation_id is not None and (
            "/compaction/" in record.operation_id or record.operation_id.endswith("/summary")
        )
        match = {
            "memory_start": record.kind == "boundary_recorded" and record.payload["stage"] == "memory_started",
            "intent": record.kind == "op_completed"
            and record.payload.get("result", {}).get("metadata", {}).get("request_compaction"),
            "plan": record.kind == "op_planned" and is_summary,
            "start": record.kind == "op_started" and is_summary,
            "receipt": record.kind == "op_completed" and is_summary,
            "replacement": record.kind == "context_compacted",
            "telemetry": record.kind == "boundary_recorded" and record.payload["stage"] == "memory_completed",
            "admission": record.kind == "op_completed" and record.payload.get("result", {}).get("tool_call_id") == "request",
        }[point]
        if match:
            crashed.append(record)
            raise Crash

    start(store)
    rt = configured(database, [call("child_task" if child else "compact"), summary, LLMResponse("done")], child=child, hook=hook)
    with pytest.raises(Crash):
        drive(store, "s", runtime=rt)
    assert len(crashed) == 1
    plans_before = {r.record_id: digest(r.payload) for r in records_of(store, "op_planned")}
    # An op_started without a receipt is Unknown, and may make one risk-marked retry.
    remaining = [summary, LLMResponse("done")] if point in {"intent", "memory_start", "plan", "start"} else [LLMResponse("done")]
    drive(store, "s", runtime=configured(database, remaining, child=child))
    drive(store, "s", runtime=configured(database, [], child=child))
    assert len(calls) == 1
    assert all(
        digest(r.payload) == plans_before[r.record_id] for r in records_of(store, "op_planned") if r.record_id in plans_before
    )
    receipts = [r for r in records_of(store, "op_completed") if r.payload["usage"].get("prompt_tokens") == 12000]
    assert len(receipts) == 1
    tools = [r for r in records_of(store, "op_completed") if r.payload["result"].get("tool_call_id") == "request"]
    assert len(tools) == 1
    if child:
        assert "Compressed Agent Memory" in store.peek_inbox("child")[0].item.payload["content"]
        assert len(records_of(store, "op_parked")) == 1
    else:
        assert len(records_of(store, "context_compacted")) == 1


@pytest.mark.parametrize("child", [False, True])
def test_summary_endpoint_attempts_are_owned_by_kernel(store, database, monkeypatch, child):
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient

    calls = []

    def complete(client, request):
        assert len(client.endpoint_targets) == client.max_retries_per_endpoint == 1
        assert request.model_settings.retry.max_attempts == 1
        endpoint = client.endpoint_targets[0].endpoint_id
        calls.append((request.metadata["purpose"], endpoint, request.metadata["attempt"]))
        if request.metadata["purpose"] == "compaction":
            if endpoint == "first":
                raise TimeoutError("receipt unavailable")
            return LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})
        if len([p for p, _, _ in calls if p == "primary"]) == 1:
            return call("child_task" if child else "compact")
        return LLMResponse("done")

    monkeypatch.setattr(VvLlmClient, "complete", complete)
    start(store)
    rt = configured(database, [], child=child)
    rt.llm = VvLlmClient(
        [EndpointTarget(e, "unused", "https://example.invalid") for e in ("first", "second")],
        backend="openai",
        randomize_endpoints=False,
    )
    rt.resolved = replace(rt.resolved, backend="openai")
    drive(store, "s", runtime=rt)
    assert [(e, n) for p, e, n in calls if p == "compaction"] == [("first", 1), ("second", 2)]
    assert len(records_of(store, "op_unknown")) == 1
    assert records_of(store, "turn_ended")[-1].payload["status"] == "completed"


def process_summary_worker(database, pipe, child, cut):
    from .conftest import open_store

    def summary(request):
        with open_store(database) as counter:
            counter.connection.execute("INSERT INTO summary_calls VALUES (1)")
        return LLMResponse(SUMMARY, raw={"usage": MODEL_USAGE})

    def hook(point, record):
        receipt = record.kind == "op_completed" and record.payload.get("usage", {}).get("prompt_tokens") == 12000
        admission = child and record.kind == "op_completed" and record.payload.get("result", {}).get("tool_call_id") == "request"
        replacement = not child and record.kind == "context_compacted"
        if (cut == "receipt" and point == "after_commit" and receipt) or (
            cut == "acceptance_rollback" and point == "before_commit" and (admission or replacement)
        ):
            pipe.send(("barrier", None))
            pipe.recv()

    with open_store(database) as store:
        drive(
            store,
            "s",
            runtime=configured(database, [call("child_task" if child else "compact"), summary], child=child, hook=hook),
        )


@pytest.mark.persistent_store
@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("cut", ["receipt", "acceptance_rollback"])
def test_process_kill_reuses_paid_receipt_and_atomic_admission(store, database, child, cut):
    import multiprocessing

    from .test_recovery_matrix import kill, receive

    start(store)
    store.connection.execute("CREATE TABLE summary_calls (n integer)")
    ctx = multiprocessing.get_context("spawn")
    pipe, worker_pipe = ctx.Pipe()
    worker = ctx.Process(target=process_summary_worker, args=(database, worker_pipe, child, cut))
    worker.start()
    worker_pipe.close()
    try:
        receive(pipe, "barrier")
        kill(worker, pipe, database)
        assert not records_of(store, "op_parked")
        drive(store, "s", runtime=configured(database, [LLMResponse("done")], child=child))
        assert store.connection.execute("SELECT count(*) FROM summary_calls").fetchone()[0] == 1
        assert len(summary_plans(store)) == 1
        assert len([r for r in records_of(store, "op_completed") if r.payload["usage"].get("prompt_tokens") == 12000]) == 1
        assert_summary_billed_once(store)
        if child:
            assert len(records_of(store, "op_parked")) == 1
            assert "Compressed Agent Memory" in store.peek_inbox("child")[0].item.payload["content"]
        else:
            assert len(records_of(store, "context_compacted")) == 1
    finally:
        if worker.is_alive():
            kill(worker, pipe, database)


def test_canonical_summary_fixture_matches_real_producers(monkeypatch):
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))
    from scripts.host_summary_fixtures import produce

    fixture = Path(__file__).parents[1] / "fixtures/parity/host_summary_requests.json"
    assert produce() == json.loads(fixture.read_text())


@pytest.mark.parametrize("value", [False, None, "true", {}, 1])
def test_compaction_intent_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="request_compaction must be true"):
        ToolExecutionResult("call", "intent", metadata={"request_compaction": value})


@pytest.mark.parametrize("delivery", ["query", "callback", "late_callback"])
def test_provider_compaction_intent_and_callback_replay(store, database, delivery):
    from vv_agent.session.providers import Accepted, Definitive, Unknown
    from vv_agent.session.records import InboxItem, Record

    class Provider:
        handle = None
        plan: Record | None = None

        def submit(self, plan, *, context):
            self.plan = plan
            self.handle = {
                "kind": "provider",
                "provider": "compact",
                "job_id": "job",
                "operation_id": plan.operation_id,
                "attempt": plan.attempt,
                "request_digest": plan.payload["request_digest"],
                "evidence": "receipt",
                "query_ref": "job",
                "cancel_ref": None,
            }
            return Unknown("unconfirmed") if delivery == "late_callback" else Accepted(self.handle)

        def query(self, handle):
            if delivery == "query":
                return Definitive(result, ("receipt",))
            return Accepted(handle)

        def authenticate(self, item, plan):
            return item.payload["evidence"] == ["receipt"]

    # Use an explicit result, so the provider never invokes a function-tool adapter.
    result = ToolExecutionResult("request", "requested", metadata={"request_compaction": True}).to_dict()
    provider = Provider()
    start(store)
    steps = [call(), LLMResponse(SUMMARY), LLMResponse("done")] if delivery != "late_callback" else [call(), LLMResponse("done")]
    rt = configured(database, steps)
    rt.providers["compact"] = provider
    rt.poll_ms = 1
    drive(store, "s", runtime=rt)
    if delivery == "query":
        from threading import Event

        Event().wait(0.01)
        drive(store, "s", runtime=rt)
        assert len(records_of(store, "context_compacted")) == 1
        receipt = next(r for r in records_of(store, "op_completed") if r.payload["result"].get("tool_call_id") == "request")
        assert "compaction_source" in receipt.payload["result"]["metadata"]
        return
    plan = provider.plan
    assert plan is not None
    for index in range(2):
        with store.atomic() as tx:
            tx.push(
                "s",
                InboxItem(
                    f"callback-{index}",
                    "provider_result",
                    {
                        "operation_id": plan.operation_id,
                        "attempt": plan.attempt,
                        "request_digest": plan.payload["request_digest"],
                        "provider_binding": "compact",
                        "result": result,
                        "evidence": ["receipt"],
                        "usage": {},
                    },
                    plan.turn_id,
                    generation=0,
                ),
            )
        drive(store, "s", runtime=rt)
    applied = [r for r in records_of(store, "input_applied") if r.payload["input"]["input_id"].startswith("callback-")]
    assert [r.payload["disposition"] for r in applied] == ["applied", "noop"]
    assert len(records_of(store, "context_compacted")) == (0 if delivery == "late_callback" else 1)


def test_child_summary_does_not_consume_unknown_outside_its_source(store, database):
    from vv_agent.session.providers import Unknown

    @function_tool
    def effect() -> str:
        raise AssertionError("provider owns execution")

    class EffectProvider:
        def submit(self, plan, *, context):
            return Unknown("effect has no receipt")

    observed = []

    def hook(point, record):
        if point == "after_commit" and record.kind == "op_planned" and record.payload["purpose"] == "compaction":
            observed.append(record.payload["consumed_unknowns"])

    start(store)
    rt = configured(
        database,
        [
            LLMResponse("", [ToolCall("effect", "effect", {}), ToolCall("request", "child_task", {})]),
            LLMResponse(SUMMARY),
            LLMResponse("done"),
        ],
        child=True,
        hook=hook,
    )
    rt.agent.tools.append(effect)
    rt.providers["effect"] = EffectProvider()
    drive(store, "s", runtime=rt)
    assert observed == [[]]
    primary = [r for r in records_of(store, "op_planned") if r.payload["purpose"] == "primary"]
    assert len(primary[-1].payload["consumed_unknowns"]) == 1
    assert primary[-1].payload["consumed_unknowns"][0]["operation_id"].endswith("/tool/0")
    assert not records_of(store, "context_compacted")
