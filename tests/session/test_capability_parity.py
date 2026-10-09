"""F2d-2 paired producers and recovery boundaries for memory, budgets and projections."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from support import FixedModelProvider

from vv_agent import Agent, RunConfig, Runner
from vv_agent.budget import HostCost, RunBudgetLimits, UnavailableMetricPolicy
from vv_agent.events import DiagnosticEvent, ToolCallPlannedEvent, ToolCallStartedEvent, event_from_dict
from vv_agent.llm.scripted import ScriptedLLM, ScriptStep
from vv_agent.memory import MemoryManager
from vv_agent.memory.provider import MemoryProviderResult
from vv_agent.output_validation import OutputValidationResult
from vv_agent.runtime.hooks import BaseRuntimeHook
from vv_agent.runtime.lifecycle import AfterCycleDecision
from vv_agent.session.events import SessionRunEventStore
from vv_agent.session.kernel import drive
from vv_agent.session.result import project_result
from vv_agent.session.tracing import deliver_spans
from vv_agent.types import AgentStatus, LLMResponse, Message, ToolCall

from .test_runner_parity import RESOLVED, echo
from .test_tools_control_parity import Restart, admit, records, runtime

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "prompt_tokens_details": {"cached_tokens": 4}}
SUMMARY = json.dumps({"original_user_messages": ["request"], "current_work_state": "Continue"})


def old_run(agent, config, steps):
    return Runner.run_sync(
        agent, "go", run_config=replace(config, model_provider=FixedModelProvider(ScriptedLLM(deepcopy(steps)), RESOLVED))
    )


def new_run(store, database, agent, config, steps, *, memory=None, cut=None):
    admit(store, config)
    llm = ScriptedLLM(deepcopy(steps))
    rt = runtime(
        store, database, agent, config, llm, **({"memory_manager": memory} if memory else {}), **({"hook": cut} if cut else {})
    )
    if cut:
        with pytest.raises(Restart):
            drive(store, "control", runtime=rt)
        store._fold_cache = None
        rt = runtime(store, database, agent, config, llm, **({"memory_manager": memory} if memory else {}))
    drive(store, "control", runtime=rt)
    return project_result(store, "control", "control/turn/initial", runtime=rt)


def same_result(new, old, *, budget=True):
    for field in ("final_output", "status", "completion_reason", "completion_tool_name", "partial_output", "wait_reason"):
        assert getattr(new, field) == getattr(old, field), field
    assert new.raw_result.error == old.raw_result.error
    if budget:
        for name in ("cycles", "total_tokens", "uncached_input_tokens", "tool_calls", "tool_calls_by_name"):
            if new.budget_usage or old.budget_usage:
                assert getattr(new.budget_usage, name) == getattr(old.budget_usage, name), name


def cut_boundary(stage):
    def cut(point, record):
        if point == "after_commit" and record.kind == "boundary_recorded" and record.payload["stage"] == stage:
            raise Restart

    return cut


def test_before_memory_hook_replacement_restart_parity(store, database, tmp_path):
    seen = []

    class Hook(BaseRuntimeHook):
        def before_memory_compact(self, event):
            seen.append(event.cycle_index)
            event.shared_state["memory_hook"] = "saved"
            return [*event.messages, Message("user", "persisted patch")]

    def answer(request):
        assert request.messages[-1].content == "persisted patch"
        return LLMResponse("done")

    agent = Agent("test", "test", hooks=[Hook()])
    config = RunConfig(workspace=tmp_path)
    old = old_run(agent, config, [answer])
    new = new_run(store, database, agent, config, [answer], cut=cut_boundary("before_memory"))
    same_result(new, old)
    assert seen == [1, 1]
    assert new.raw_result.shared_state == old.raw_result.shared_state
    assert len([r for r in records(store, kind="boundary_recorded") if r.payload["stage"] == "before_memory"]) == 1


@pytest.mark.parametrize("action", ["continue", "steer", "deny", "stop", "invalid", "raise"])
def test_after_cycle_decision_snapshot_restart_parity(store, database, tmp_path, action):
    def setup():
        snapshots = []

        class Hook:
            def after_cycle(self, snapshot):
                snapshots.append(snapshot)
                assert snapshot.cycle.index == snapshot.cycle_index
                if snapshot.cycle_index == 1:
                    assert snapshot.cycle.tool_results[0].content == "one"
                assert snapshot.shared_state["tenant"] == "test"
                if action == "raise":
                    raise RuntimeError("host failed")
                if action == "invalid":
                    return "invalid"
                if action == "stop":
                    return AfterCycleDecision.stop_non_success(code="host_stop", message="stop")
                if snapshot.cycle_index == 1:
                    if action == "steer":
                        return AfterCycleDecision.steer(["review once"], disallow_tools=["echo"])
                    if action == "deny":
                        return AfterCycleDecision.continue_run(disallow_tools=["echo"])
                return None

        return Hook(), snapshots

    def answer(request):
        if action in {"steer", "deny"}:
            assert "echo" not in [s["function"]["name"] for s in request.tools]
        if action == "steer":
            assert any(m.content == "review once" for m in request.messages)
        return LLMResponse("done", raw={"usage": USAGE})

    # The hook's second snapshot also needs a tool result; stop checking its first-cycle facts then.
    steps = [LLMResponse("working", [ToolCall("a", "echo", {"text": "one"})], raw={"usage": USAGE}), answer]
    hook, old_snapshots = setup()
    config = RunConfig(workspace=tmp_path, shared_state={"tenant": "test"}, after_cycle_hooks=[hook])
    agent = Agent("test", "test", tools=[echo])
    old = old_run(agent, config, steps)
    hook, new_snapshots = setup()
    config = replace(config, after_cycle_hooks=[hook])
    new = new_run(store, database, agent, config, steps, cut=cut_boundary("after_cycle"))
    same_result(new, old)
    assert len(new_snapshots) == len(old_snapshots)
    for a, b in zip(new_snapshots, old_snapshots, strict=True):
        assert a.cycle.to_dict() == b.cycle.to_dict()
        assert a.native_outcome == b.native_outcome
        assert a.cumulative_token_usage == b.cumulative_token_usage
    decided = [r for r in records(store, kind="boundary_recorded") if r.payload["stage"] == "after_cycle"]
    assert len(decided) == len(new_snapshots)


@pytest.mark.parametrize("wait", [False, True])
def test_after_cycle_steering_boundary_parity(store, database, tmp_path, wait):
    class Hook:
        def after_cycle(self, snapshot):
            return AfterCycleDecision.steer(["again"])

    config = RunConfig(
        workspace=tmp_path, after_cycle_hooks=[Hook()], max_cycles=1, no_tool_policy="wait_user" if wait else "finish"
    )
    agent = Agent("test", "test")
    old = old_run(agent, config, [LLMResponse("first")])
    new = new_run(store, database, agent, config, [LLMResponse("first")])
    same_result(new, old)
    assert "after_cycle_steer_unavailable" in new.raw_result.error["message"]


class MemoryCallbacks:
    def __init__(self):
        self.calls = []

    def before_compact(self, event):
        self.calls.append(("before", event))
        assert event.metadata["messages"]
        return MemoryProviderResult({"marker": "retained"})

    def after_compact(self, event):
        self.calls.append(("after", event))


def memory_config(tmp_path):
    history = [Message("user", "request"), Message("assistant", "old facts " * 4000)]
    return RunConfig(
        workspace=tmp_path,
        initial_messages=history,
        metadata={"memory_compact_threshold": 1000, "memory_keep_recent_messages": 1},
    )


def paired_memory_manager(monkeypatch, threshold=1000, keep=1):
    from vv_agent.runtime.engine import AgentRuntime

    original = AgentRuntime._build_memory_manager

    def build(self, **kwargs):
        manager = original(self, **kwargs)
        manager.compact_threshold = threshold
        manager.keep_recent_messages = keep
        manager.model = ""
        return manager

    monkeypatch.setattr(AgentRuntime, "_build_memory_manager", build)
    return MemoryManager(compact_threshold=threshold, keep_recent_messages=keep)


@pytest.mark.parametrize("stage", ["memory_started", "memory_completed"])
@pytest.mark.parametrize("accepted", [True, False])
def test_memory_provider_logged_callbacks_restart_parity(store, database, tmp_path, monkeypatch, stage, accepted):
    memory = paired_memory_manager(monkeypatch)
    agent = Agent("test", "test")
    callbacks = MemoryCallbacks()
    config = replace(memory_config(tmp_path), memory_providers=[callbacks])
    steps = [LLMResponse(SUMMARY if accepted else "bad summary", raw={"usage": USAGE}), LLMResponse("done", raw={"usage": USAGE})]
    old = old_run(agent, config, steps)
    old_calls = callbacks.calls
    callbacks = MemoryCallbacks()
    new = new_run(
        store, database, agent, replace(config, memory_providers=[callbacks]), steps, memory=memory, cut=cut_boundary(stage)
    )
    same_result(new, old)
    assert [name for name, _ in callbacks.calls] == [name for name, _ in old_calls] == ["before", "after"]
    for (_, a), (_, b) in zip(callbacks.calls, old_calls, strict=True):
        for field in ("type", "cycle_index"):
            assert getattr(a, field) == getattr(b, field)
    for event in new.events:
        assert event_from_dict(event.to_dict(), _kernel=True).to_dict() == event.to_dict()
    lifecycle = [e for e in new.events if e.type.startswith("memory_compact_")]
    assert len(lifecycle) == 2
    for a, b in zip(lifecycle, [e for e in old.events if e.type.startswith("memory_compact_")], strict=True):
        fields = (
            ("message_count", "estimated_tokens", "trigger", "candidate_count")
            if a.type.endswith("started")
            else ("before_count", "after_count", "mode", "changed", "archived_count", "artifact_failure_count")
        )
        for field in fields:
            assert getattr(a, field) == getattr(b, field), field
    old_start = next(e for e in old.events if e.type == "memory_compact_started")
    assert lifecycle[0].metadata["memory_provider_results"] == old_start.metadata["memory_provider_results"]
    assert lifecycle[1].changed == accepted


@pytest.mark.parametrize("mode", ["summary", "micro", "prompt_too_long"])
def test_memory_compaction_runner_producer_parity(store, database, tmp_path, monkeypatch, mode):
    threshold = 999999 if mode == "prompt_too_long" else 1000
    memory = paired_memory_manager(monkeypatch, threshold)
    config = memory_config(tmp_path)
    if mode == "micro":
        from vv_agent.microcompaction import MicrocompactionPolicy

        history = [
            Message("user", "request"),
            Message(
                "assistant",
                "",
                tool_calls=[
                    {"id": "old", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"old.txt"}'}}
                ],
            ),
            Message("tool", "old result " * 5000, tool_call_id="old", name="read_file"),
            Message("assistant", "recent"),
        ]
        memory.compact_threshold = 999999
        # Trigger by context pressure while preserving a recoverable raw tail.
        memory.microcompaction_policy = MicrocompactionPolicy(keep_recent_cycles=1, trigger_ratio=0.0001, target_ratio=0.00005)
        config = replace(config, initial_messages=history, microcompaction_policy=memory.microcompaction_policy)
        from vv_agent.runtime.engine import AgentRuntime

        original = AgentRuntime._build_memory_manager

        def micro(self, **kwargs):
            manager = original(self, **kwargs)
            manager.compact_threshold = 999999
            manager.microcompaction_policy = memory.microcompaction_policy
            return manager

        monkeypatch.setattr(AgentRuntime, "_build_memory_manager", micro)

    def too_long(_request):
        raise RuntimeError("maximum context length exceeded")

    steps = (
        ([too_long] if mode == "prompt_too_long" else [])
        + ([] if mode == "micro" else [LLMResponse(SUMMARY)])
        + [LLMResponse("done")]
    )
    agent = Agent("test", "test")
    old = old_run(agent, config, steps)
    new = new_run(store, database, agent, config, steps, memory=memory)
    same_result(new, old)
    assert any(r.kind == "context_compacted" for r in records(store))

    def semantic(messages):
        return [(m.role, m.name, m.content if m.role != "tool" else m.content.split("artifact_path")[0]) for m in messages]

    assert semantic(new.raw_result.messages) == semantic(old.raw_result.messages)
    assert new.raw_cycles[-1].memory_compacted == old.raw_cycles[-1].memory_compacted


@pytest.mark.parametrize("metric", ["total", "uncached"])
@pytest.mark.parametrize("limit", [0, 5, 6, 7, 9, 10, 14, 15, 16, 100])
def test_token_budget_boundaries_parity(store, database, tmp_path, metric, limit):
    config = RunConfig(
        workspace=tmp_path,
        budget_limits=RunBudgetLimits(**{"max_total_tokens" if metric == "total" else "max_uncached_input_tokens": limit}),
    )
    agent = Agent("test", "test")
    steps = [LLMResponse("done", raw={"usage": USAGE})]
    same_result(new_run(store, database, agent, config, steps), old_run(agent, config, steps))


@pytest.mark.parametrize("policy", list(UnavailableMetricPolicy))
@pytest.mark.parametrize("usage", [None, {"prompt_tokens": 10, "completion_tokens": 5}, USAGE])
def test_missing_usage_budget_parity(store, database, tmp_path, policy, usage):
    config = RunConfig(
        workspace=tmp_path,
        budget_limits=RunBudgetLimits(max_total_tokens=100, max_uncached_input_tokens=100, unavailable_metric_policy=policy),
    )
    steps = [LLMResponse("done", raw={"usage": usage} if usage is not None else {})]
    agent = Agent("test", "test")
    new, old = new_run(store, database, agent, config, steps), old_run(agent, config, steps)
    same_result(new, old)
    assert [(u.dimension, u.reason) for u in new.budget_usage.unavailable_dimensions] == [
        (u.dimension, u.reason) for u in old.budget_usage.unavailable_dimensions
    ]


@pytest.mark.parametrize("per_name", [False, True])
@pytest.mark.parametrize("limit", [0, 1, 2, 3])
def test_tool_batch_admission_restart_parity(store, database, tmp_path, per_name, limit):
    config = RunConfig(
        workspace=tmp_path,
        budget_limits=RunBudgetLimits(max_tool_calls_by_name={"echo": limit})
        if per_name
        else RunBudgetLimits(max_tool_calls=limit),
    )
    agent = Agent("test", "test", tools=[echo])
    steps = [
        LLMResponse(
            "batch", [ToolCall("a", "echo", {"text": "one"}), ToolCall("b", "echo", {"text": "two"})], raw={"usage": USAGE}
        ),
        LLMResponse("done", raw={"usage": USAGE}),
    ]

    def cut(point, r):
        if (
            point == "after_commit"
            and r.kind in {"op_planned", "boundary_recorded"}
            and (
                (r.kind == "op_planned" and r.payload["op_kind"] != "model")
                or (r.kind == "boundary_recorded" and r.payload["boundary_id"].endswith("/tool_batch"))
            )
        ):
            raise Restart

    new = new_run(store, database, agent, config, steps, cut=cut)
    old = old_run(agent, config, steps)
    same_result(new, old)
    starts = [r for r in records(store, kind="op_started") if "/tool/" in r.operation_id]
    assert len(starts) == (2 if limit >= 2 else 0)
    assert new.budget_usage.tool_calls == (2 if limit >= 2 else 0)


@pytest.mark.parametrize(
    "reading", [None, HostCost("credits", 0), HostCost("credits", 10), HostCost("credits", 11), HostCost("wrong", 1)]
)
@pytest.mark.parametrize("policy", list(UnavailableMetricPolicy))
def test_host_cost_and_unavailable_metrics_parity(store, database, tmp_path, reading, policy):
    class Meter:
        def read(self):
            return reading

    config = RunConfig(
        workspace=tmp_path,
        host_cost_meter=Meter(),
        budget_limits=RunBudgetLimits(max_host_cost=HostCost("credits", 10), unavailable_metric_policy=policy),
    )
    agent = Agent("test", "test")
    steps = [LLMResponse("done", raw={"usage": USAGE})]
    new, old = new_run(store, database, agent, config, steps), old_run(agent, config, steps)
    same_result(new, old)
    assert new.budget_usage.host_cost == old.budget_usage.host_cost
    assert [(u.dimension, u.reason) for u in new.budget_usage.unavailable_dimensions] == [
        (u.dimension, u.reason) for u in old.budget_usage.unavailable_dimensions
    ]


@pytest.mark.parametrize("limit", [0, 100000])
def test_wall_time_budget_parity(store, database, tmp_path, limit):
    config = RunConfig(workspace=tmp_path, budget_limits=RunBudgetLimits(max_wall_time_ms=limit))
    agent = Agent("test", "test")
    same_result(
        new_run(store, database, agent, config, [LLMResponse("done", raw={"usage": USAGE})]),
        old_run(agent, config, [LLMResponse("done", raw={"usage": USAGE})]),
    )


@pytest.mark.parametrize("policy", list(UnavailableMetricPolicy))
def test_lost_active_interval_is_unavailable_after_restart(store, database, tmp_path, policy):
    from vv_agent.budget import BudgetDimension, BudgetExhaustionReason, BudgetUnavailableReason

    agent = Agent("test", "test")
    config = RunConfig(
        workspace=tmp_path,
        budget_limits=RunBudgetLimits(max_wall_time_ms=100000, unavailable_metric_policy=policy),
    )
    steps = [LLMResponse("done", raw={"usage": USAGE})]
    old = old_run(agent, config, steps)

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_started":
            raise Restart

    new = new_run(store, database, agent, config, steps, cut=cut)
    assert old.status == AgentStatus.COMPLETED
    assert new.status == (AgentStatus.FAILED if policy == UnavailableMetricPolicy.STOP else AgentStatus.COMPLETED)
    assert any(
        value.dimension == BudgetDimension.WALL_TIME and value.reason == BudgetUnavailableReason.ACCOUNTING_MISSING
        for value in new.budget_usage.unavailable_dimensions
    )
    assert any(r.payload["observation"]["active_interval_missing"] for r in records(store, kind="op_unknown"))
    if policy == UnavailableMetricPolicy.STOP:
        assert new.budget_exhaustion.dimension == BudgetDimension.WALL_TIME
        assert new.budget_exhaustion.reason == BudgetExhaustionReason.METRIC_UNAVAILABLE
        assert len(records(store, kind="op_started")) == 1
    else:
        assert new.final_output == old.final_output
        assert len(records(store, kind="op_started")) == 2
    store._fold_cache = None
    rt = runtime(store, database, agent, config, ScriptedLLM([]))
    rebuilt = project_result(store, "control", new.run_id, runtime=rt)
    assert rebuilt.budget_usage == new.budget_usage and rebuilt.budget_exhaustion == new.budget_exhaustion


@dataclass
class Answer:
    value: int


@pytest.mark.parametrize("output_type", [dict, list, Answer])
@pytest.mark.parametrize("repair", [False, True])
def test_typed_output_repair_ledger_restart_parity(store, database, tmp_path, output_type, repair):
    calls = []
    good = "[1, 2]" if output_type is list else '{"value": 1}'

    def fix(request):
        calls.append(request)
        assert request.tools == ()
        return good

    agent = Agent(
        "test",
        "test",
        output_type=output_type,
        output_validation_enabled=True,
        output_validator=lambda value, ctx: OutputValidationResult.accept(),
        output_repair=fix if repair else None,
    )
    config = RunConfig(workspace=tmp_path)
    if output_type is Answer and repair:
        with pytest.raises(TypeError, match="not JSON serializable"):
            old_run(agent, config, [LLMResponse("bad")])
        old = None
    else:
        old = old_run(agent, config, [LLMResponse("bad" if repair else good)])
    new = new_run(store, database, agent, config, [LLMResponse("bad" if repair else good)], cut=cut_boundary("output_checked"))
    if old is not None:
        same_result(new, old)
    else:
        assert new.final_output == Answer(1) and new.status == AgentStatus.COMPLETED
    ledger = new.metadata["session_model_calls"]
    from vv_agent.types import ModelCallRecord, TaskTokenUsage

    usage_wire = new.token_usage.to_dict()
    assert usage_wire["schema_version"] == "vv-agent.task-token-usage.v3"
    assert TaskTokenUsage.from_dict(usage_wire, _kernel=True).to_dict() == usage_wire
    with pytest.raises(ValueError):
        TaskTokenUsage.from_dict(usage_wire | {"schema_version": "vv-agent.task-token-usage.v2"}, _kernel=True)
    for call in new.token_usage.model_calls:
        wire = call.to_dict()
        assert wire["schema_version"] == "vv-agent.model-call.v2"
        assert wire["usage"]["schema_version"] == "vv-agent.token-usage.v1"
        assert ModelCallRecord.from_dict(wire, _kernel=True).to_dict() == wire
        with pytest.raises(ValueError):
            ModelCallRecord.from_dict(wire | {"schema_version": "vv-agent.model-call.v1"}, _kernel=True)
    assert [call["purpose"] for call in ledger] == (["primary", "output_repair"] if repair else ["primary"])
    assert len(calls) == (2 if repair else 0)
    if repair:
        assert len(new.token_usage.model_calls) == 2
        assert new.token_usage.model_calls[-1].operation.value == "output_repair"
        assert ledger[-1]["usage"]["usage_source"] == "accounting_missing"


@pytest.mark.parametrize("failure", ["repair_exception", "validator_exception", "validator_contract", "repair_invalid"])
def test_output_repair_exceptions_parity(store, database, tmp_path, failure):
    def validator(value, ctx):
        if failure == "validator_exception":
            raise RuntimeError("validator failed")
        if failure == "validator_contract":
            return True
        return OutputValidationResult.reject("invalid", "bad")

    def fix(request):
        if failure == "repair_exception":
            raise RuntimeError("repair failed")
        return "still invalid"

    agent = Agent("test", "test", output_validation_enabled=True, output_validator=validator, output_repair=fix)
    config = RunConfig(workspace=tmp_path)
    new, old = new_run(store, database, agent, config, [LLMResponse("bad")]), old_run(agent, config, [LLMResponse("bad")])
    assert new.final_output == old.final_output
    assert new.status == old.status
    assert new.raw_result.error == old.raw_result.error
    assert new.partial_output == old.partial_output


def test_trace_delivery_span_parity_and_no_recovery_duplicates(store, database, tmp_path):
    class Processor:
        def __init__(self):
            self.spans = []

        def on_span_start(self, span):
            self.spans.append(("start", span))

        def on_span_end(self, span):
            self.spans.append(("end", span))

    old_processor, new_processor = Processor(), Processor()
    agent = Agent("test", "test", tools=[echo])
    config = RunConfig(workspace=tmp_path, tracing={"processors": [old_processor]})
    steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", "echo", {"text": "one"})]), LLMResponse("done")]
    old_run(agent, config, steps)
    # The internal assembly registers the trace cursor with session creation.
    from vv_agent.session.records import InboxItem, SessionSpec

    with store.atomic() as tx:
        tx.create(SessionSpec("control", "test", str(tmp_path)), consumers=("events", "traces"))
        tx.push("control", InboxItem("initial", "user", {"content": "go"}))
    rt = runtime(store, database, agent, config, ScriptedLLM(deepcopy(steps)))
    drive(store, "control", runtime=rt)
    assert deliver_spans(store, "control", [new_processor]) == 6
    store._fold_cache = None
    assert deliver_spans(store, "control", [new_processor]) == 0
    assert [(stage, span.name) for stage, span in new_processor.spans] == [
        (stage, span.name) for stage, span in old_processor.spans
    ]
    starts = {s.span_id: s for action, s in new_processor.spans if action == "start"}
    assert len(starts) == 3
    assert all(s.span_id in starts and s.ended_at >= s.started_at for action, s in new_processor.spans if action == "end")


def test_lifecycle_events_replay_ack_and_rollback_parity(store, database, tmp_path):
    agent = Agent("test", "test", tools=[echo])
    config = RunConfig(workspace=tmp_path)
    steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", "echo", {"text": "one"})]), LLMResponse("done")]
    old = old_run(agent, config, steps)
    new = new_run(store, database, agent, config, steps)
    for kind in ("agent_started", "cycle_started", "diagnostic"):
        assert [e.type for e in new.events].count(kind) == [e.type for e in old.events].count(kind)
    bridge = SessionRunEventStore(store, "control")
    replayed = list(bridge.replay(run_id=new.run_id))
    assert [e.to_dict() for e in replayed] == [e.to_dict() for e in new.events]
    assert len({e.event_id for e in replayed}) == len(replayed)
    bridge.append(replayed[0])
    with pytest.raises(ValueError):
        bridge.append(event_from_dict(replayed[0].to_dict() | {"event_id": "external"}, _kernel=True))
    with pytest.raises(Restart), store.atomic() as tx:
        result = bridge.batch(tx)
        assert result is not None
        batch, _ = result
        tx.ack(batch)
        raise Restart
    delivered = []
    while bridge.consume(delivered.append, limit=1):
        pass
    assert [e.event_id for e in delivered] == [e.event_id for e in replayed]
    store._fold_cache = None
    assert SessionRunEventStore(store, "control").consume(delivered.append) == 0
    assert list(bridge.replay(run_id=new.run_id)) == replayed


def test_session_memory_logged_extract_save_reload_parity(store, database, tmp_path, monkeypatch):
    memory = paired_memory_manager(monkeypatch, 999999)
    meta = {"session_id": "remember", "session_memory_min_tokens": 1, "session_memory_min_text_messages": 1}
    agent = Agent("test", "test")
    extraction = '[{"category":"user_intent","content":"retain user goal","importance":8}]'
    steps = [LLMResponse(extraction, raw={"usage": USAGE}), LLMResponse("done", raw={"usage": USAGE})]
    old_config = RunConfig(workspace=tmp_path / "old", session_memory_enabled=True, metadata=meta)
    assert isinstance(old_config.workspace, Path)
    old_config.workspace.mkdir()
    old = old_run(agent, old_config, steps)
    config = replace(old_config, workspace=tmp_path / "new")
    assert isinstance(config.workspace, Path)
    config.workspace.mkdir()
    new = new_run(store, database, agent, config, steps, memory=memory, cut=cut_boundary("session_memory_saved"))
    same_result(new, old)
    assert [c["purpose"] for c in new.metadata["session_model_calls"]] == ["session_memory", "primary"]
    old_file = Path(old_config.workspace) / ".memory/session/remember/session_memory.json"
    new_file = Path(config.workspace) / ".memory/session/remember/session_memory.json"
    assert json.loads(new_file.read_text()) == json.loads(old_file.read_text())
    # Rebuild the disposable file entirely from log state before compiling the next turn.
    new_file.unlink()
    from vv_agent.session.records import InboxItem

    with store.atomic() as tx:
        tx.push("control", InboxItem("second", "user", {"content": "go"}))
    requests = []

    def answer(request):
        requests.append(request)
        assert "retain user goal" in request.prompt_bundle.flatten()
        return LLMResponse("reloaded")

    config2 = replace(config, metadata=meta | {"session_memory_min_tokens": 999999})
    old2 = old_run(agent, replace(old_config, metadata=config2.metadata), [answer])
    rt = runtime(store, database, agent, config2, ScriptedLLM([answer]), memory_manager=memory)
    store._fold_cache = None
    drive(store, "control", runtime=rt)
    new2 = project_result(store, "control", "control/turn/second", runtime=rt)
    same_result(new2, old2)
    assert len(requests) == 2 and new_file.exists()


class StreamingScripted(ScriptedLLM):
    def complete_with_stream(self, request, stream_callback=None):
        response = self.complete(request)
        if stream_callback:
            for payload in [
                {"event": "assistant_delta", "content_delta": "volatile"},
                {"event": "reasoning_delta", "reasoning_delta": "consider"},
                {"event": "tool_call_started", "tool_call_id": "a", "function_name": "echo", "tool_call_index": 0},
                {"event": "tool_call_progress", "tool_call_id": "a", "function_name": "echo", "arguments_chars": 4},
                {"event": "unknown"},
                {"event": "assistant_delta", "content_delta": 4},
            ]:
                stream_callback(payload)
        return response


@pytest.mark.parametrize("drop", [False, True])
def test_live_stream_deltas_and_durable_final_restart_parity(store, database, tmp_path, drop):
    config = RunConfig(workspace=tmp_path)
    agent = Agent("test", "test", tools=[echo])
    steps: list[ScriptStep] = [
        LLMResponse("durable content", [ToolCall("a", "echo", {"text": "one"})], raw={"reasoning_content": "durable reasoning"}),
        LLMResponse("done"),
    ]
    old_deltas, new_deltas = [], []
    old = Runner.run_sync(
        agent,
        "go",
        run_config=replace(
            config, stream=old_deltas.append, model_provider=FixedModelProvider(StreamingScripted(deepcopy(steps)), RESOLVED)
        ),
    )

    def sink(event):
        if drop:
            raise RuntimeError("volatile sink unavailable")
        new_deltas.append(event)

    admit(store, config)
    llm = StreamingScripted(deepcopy(steps))

    def cut(point, r):
        if (
            point == "after_commit"
            and r.kind == "op_planned"
            and "/primary/1/attempt/" in r.operation_id
            and r.payload["op_kind"] != "model"
        ):
            raise Restart

    rt = runtime(store, database, agent, replace(config, stream=sink), llm, hook=cut)
    with pytest.raises(Restart):
        drive(store, "control", runtime=rt)
    delivered_before = len(new_deltas)
    store._fold_cache = None
    rt = runtime(store, database, agent, replace(config, stream=sink), llm)
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    same_result(new, old)
    kinds = {"assistant_delta", "reasoning_delta", "model_tool_call_started", "model_tool_call_progress"}
    if not drop:
        assert [e.type for e in new_deltas] == [e.type for e in old_deltas if e.type in kinds]
        assert delivered_before == 4
    else:
        assert not new_deltas
    assert not kinds & {e.type for e in new.events}
    first = next(m for m in new.raw_result.messages if m.role == "assistant")
    assert first.content == "durable content" and first.reasoning_content == "durable reasoning"
    assert len([r for r in records(store, kind="op_started") if r.operation_id.endswith("/primary/1")]) == 1


@pytest.mark.parametrize("preferred", [None, "b"])
@pytest.mark.parametrize("randomized", [False, True])
@pytest.mark.parametrize("failures", [1, 2])
def test_endpoint_routing_preference_logged_attempts_parity(
    store, database, tmp_path, monkeypatch, preferred, randomized, failures
):
    from types import SimpleNamespace

    import httpx
    from vv_llm.types import APIConnectionError

    import vv_agent.llm.vv_llm_client as client_module
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
    from vv_agent.model_settings import ModelSettings, RetrySettings

    monkeypatch.setattr(client_module.random, "shuffle", lambda items: items.reverse())
    targets = [EndpointTarget(n, "unused", "https://example.invalid") for n in ("a", "b", "c")]
    calls = []
    fail_remaining = failures

    def transport(*, endpoint_id, **kwargs):
        assert kwargs["random_endpoint"] is False
        return SimpleNamespace(endpoint_id=endpoint_id)

    def response(self, *, chat_client, options, **kwargs):
        nonlocal fail_remaining
        assert options.max_attempts == 1
        calls.append(chat_client.endpoint_id)
        if fail_remaining:
            fail_remaining -= 1
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        return LLMResponse("done", raw={"usage": USAGE})

    monkeypatch.setattr(client_module, "create_chat_client", transport)
    monkeypatch.setattr(VvLlmClient, "_non_stream_completion", response)
    monkeypatch.setattr(VvLlmClient, "_should_use_stream", staticmethod(lambda _: False))
    monkeypatch.setattr(client_module, "format_messages", lambda *, messages, **kwargs: messages)

    def client():
        llm = VvLlmClient(targets, backend="openai", randomize_endpoints=randomized, backoff_seconds=0)
        llm._preferred_endpoint_id = preferred
        return llm

    reference = client()
    config = RunConfig(workspace=tmp_path, model_settings=ModelSettings(retry=RetrySettings(max_attempts=1)))
    agent = Agent("test", "test")
    old = Runner.run_sync(agent, "go", run_config=replace(config, model_provider=FixedModelProvider(reference, RESOLVED)))
    old_calls = list(calls)
    old2 = Runner.run_sync(agent, "go", run_config=replace(config, model_provider=FixedModelProvider(reference, RESOLVED)))
    old_next = calls[-1]
    calls.clear()
    fail_remaining = failures
    admit(store, config)

    def cut(point, r):
        if point == "after_commit" and r.kind == "op_planned":
            raise Restart

    initial = client()
    rt = runtime(store, database, agent, config, initial, hook=cut)
    with pytest.raises(Restart):
        drive(store, "control", runtime=rt)
    replacement = client()
    replacement._preferred_endpoint_id = "a"
    rt = runtime(store, database, agent, config, replacement)
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert (new.final_output, new.status) == (old.final_output, old.status)
    assert calls == old_calls
    assert [r.payload["endpoint_id"] for r in records(store, kind="op_started")] == calls
    planned = records(store, kind="op_planned")
    assert len({r.payload["request_digest"] for r in planned}) == 1
    assert initial.max_retries_per_endpoint == 3
    from vv_agent.session.records import InboxItem

    with store.atomic() as tx:
        tx.push("control", InboxItem("second", "user", {"content": "go"}))
    drive(store, "control", runtime=rt)
    assert calls[-1] == old_next
    new2 = project_result(store, "control", "control/turn/second", runtime=rt)
    assert (new2.final_output, new2.status) == (old2.final_output, old2.status)
    # A terminal result remains the original turn's projection after subsequent work.
    assert project_result(store, "control", new.run_id, runtime=rt).raw_result.messages == new.raw_result.messages


@pytest.mark.parametrize("wait", ["no_tool", "ask_user"])
def test_wait_result_fields_parity(store, database, tmp_path, wait):
    agent = Agent("test", "test")
    config = RunConfig(workspace=tmp_path, no_tool_policy="wait_user")
    steps = [LLMResponse("Question?", [] if wait == "no_tool" else [ToolCall("ask", "ask_user", {"question": "Question?"})])]
    old, new = old_run(agent, config, steps), new_run(store, database, agent, config, steps)
    same_result(new, old)
    assert new.status == AgentStatus.WAIT_USER and new.wait_reason


def test_repair_usage_budget_is_logged_intentional_difference(store, database, tmp_path):
    def fix(request):
        return LLMResponse("valid", raw={"usage": USAGE})

    agent = Agent(
        "test",
        "test",
        output_validation_enabled=True,
        output_validator=lambda value, ctx: (
            OutputValidationResult.accept() if value == "valid" else OutputValidationResult.reject("invalid")
        ),
        output_repair=fix,
    )
    config = RunConfig(workspace=tmp_path, budget_limits=RunBudgetLimits(max_total_tokens=20))
    # Runner treats callback LLMResponse as an arbitrary candidate and has no repair accounting.
    old = old_run(agent, config, [LLMResponse("invalid", raw={"usage": USAGE})])
    new = new_run(store, database, agent, config, [LLMResponse("invalid", raw={"usage": USAGE})])
    assert old.status == new.status == AgentStatus.FAILED
    assert old.raw_result.error["code"] == "output_validation_failed"
    assert new.completion_reason.value == "budget_exhausted"
    assert new.budget_usage.total_tokens == 30 and new.budget_usage.cycles == 1
    assert new.budget_exhaustion is not None and new.budget_exhaustion.observed == 30
    assert [c["purpose"] for c in new.metadata["session_model_calls"]] == ["primary", "output_repair"]


@pytest.mark.parametrize("boundary", ["ask_user", "finish_steer"])
def test_after_cycle_wait_and_native_finish_snapshot_restart(store, database, tmp_path, boundary):
    snapshots = []

    class Hook:
        def after_cycle(self, snapshot):
            snapshots.append(snapshot)
            if boundary == "finish_steer" and snapshot.cycle_index == 1:
                return AfterCycleDecision.steer(["check final"])
            return None

    from vv_agent.tools.function import function_tool

    @function_tool
    def finish():
        from vv_agent.types import ToolDirective, ToolExecutionResult

        return ToolExecutionResult("", "first final", directive=ToolDirective.FINISH)

    agent = Agent("test", "test", tools=[finish])
    config = RunConfig(workspace=tmp_path, after_cycle_hooks=[Hook()])
    steps = (
        [LLMResponse("question", [ToolCall("a", "ask_user", {"question": "Question?"})])]
        if boundary == "ask_user"
        else [LLMResponse("first", [ToolCall("a", "finish", {})]), LLMResponse("checked")]
    )
    old = old_run(agent, config, steps)
    old_snapshots = list(snapshots)
    snapshots.clear()
    new = new_run(store, database, agent, config, steps, cut=cut_boundary("after_cycle"))
    same_result(new, old)
    for a, b in zip(snapshots, old_snapshots, strict=True):
        assert a.cycle.to_dict() == b.cycle.to_dict()
        assert a.native_outcome == b.native_outcome
        assert [(m.role, m.content) for m in a.messages] == [(m.role, m.content) for m in b.messages]


@pytest.mark.parametrize("failure", ["raise", "currency", "decrease"])
@pytest.mark.parametrize("policy", list(UnavailableMetricPolicy))
def test_host_meter_failure_latch_restart_parity(store, database, tmp_path, failure, policy):
    class Meter:
        value = 5

        def read(self):
            if failure == "raise":
                raise RuntimeError("meter unavailable")
            return HostCost("credits", self.value, currency="EUR" if failure == "currency" else "USD")

    def run_config():
        meter = Meter()

        def answer(_request):
            meter.value = 1
            return LLMResponse("done", raw={"usage": USAGE})

        return RunConfig(
            workspace=tmp_path,
            host_cost_meter=meter,
            budget_limits=RunBudgetLimits(
                max_host_cost=HostCost("credits", 100, currency="USD"), unavailable_metric_policy=policy
            ),
        ), [answer]

    agent = Agent("test", "test")
    config, steps = run_config()
    old = old_run(agent, config, steps)
    config, steps = run_config()
    new = new_run(store, database, agent, config, steps)
    same_result(new, old)
    assert new.budget_usage.unavailable_dimensions == old.budget_usage.unavailable_dimensions
    assert new.budget_usage.host_cost == old.budget_usage.host_cost
    store._fold_cache = None
    rt = runtime(store, database, agent, config, ScriptedLLM([]))
    assert project_result(store, "control", new.run_id, runtime=rt).budget_usage == new.budget_usage


@pytest.mark.parametrize("fault", ["unknown_field", "bad_message", "missing_source", "bad_source", "bad_state", "bad_budget"])
def test_boundary_validation_rolls_back(store, database, tmp_path, fault):
    from vv_agent.session.records import make_record

    class Hook(BaseRuntimeHook):
        def before_memory_compact(self, event):
            return event.messages

    config, agent = RunConfig(workspace=tmp_path), Agent("test", "test", hooks=[Hook()])
    admit(store, config)
    with pytest.raises(Restart):
        drive(
            store, "control", runtime=runtime(store, database, agent, config, ScriptedLLM([]), hook=cut_boundary("before_memory"))
        )
    state, rows, watermark = store.read_state("control")
    before = next(r.record for r in rows if r.record.kind == "boundary_recorded")
    payload = before.payload | {"boundary_id": "invalid"}
    if fault == "unknown_field":
        payload["data"]["extension"] = True
    elif fault == "bad_message":
        payload["data"]["messages"][0]["role"] = "invalid"
    elif fault == "missing_source":
        payload["source_digest"] = None
    elif fault == "bad_source":
        payload["source_digest"] = "0" * 64
    elif fault == "bad_budget":
        from vv_agent.budget import BudgetUsageSnapshot

        payload |= {
            "stage": "budget",
            "source_digest": None,
            "data": {"usage": BudgetUsageSnapshot().to_dict() | {"unknown": True}, "exhaustion": None, "tool_names": []},
        }
    else:
        payload |= {
            "stage": "session_memory_saved",
            "source_digest": None,
            "data": {
                "state": {
                    "entries": [],
                    "initialized": "false",
                    "tokens_at_last_extraction": 0,
                    "last_extracted_message_index": -1,
                }
            },
        }
    lease = store.acquire("control", owner="validation", ttl_ms=15000)
    try:
        with pytest.raises((ValueError, TypeError)), store.atomic() as tx:
            record = make_record("boundary_recorded", session_id="control", turn_id=state.active_turn_id, payload=payload)
            tx.append(
                "control",
                lease=lease,
                expected_seq=rows[-1].seq,
                expected_inbox_seq=watermark,
                commit_id="invalid",
                records=(record,),
            )
        assert store.read_state("control")[1] == rows
    finally:
        store.release(lease)


def test_logged_endpoint_dispatch_rejects_routing_tamper(store, database, tmp_path):
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient
    from vv_agent.session.records import make_record

    agent, config = Agent("test", "test"), RunConfig(workspace=tmp_path)
    admit(store, config)
    llm = VvLlmClient([EndpointTarget(n, "unused", "https://example.invalid") for n in ("a", "b")], randomize_endpoints=False)

    def cut(point, r):
        if point == "after_commit" and r.kind == "op_planned":
            raise Restart

    with pytest.raises(Restart):
        drive(store, "control", runtime=runtime(store, database, agent, config, llm, hook=cut))
    _, rows, watermark = store.read_state("control")
    plan = next(r.record for r in rows if r.record.kind == "op_planned")
    lease = store.acquire("control", owner="validation", ttl_ms=15000)
    try:
        started = make_record(
            "op_started",
            session_id="control",
            turn_id=plan.turn_id,
            operation_id=plan.operation_id,
            attempt=plan.attempt,
            payload={
                "dispatch_id": "tamper",
                "authorization_version": "1",
                "epoch": lease.epoch,
                "mode": "sync",
                "endpoint_id": "b",
            },
        )
        with pytest.raises(ValueError, match="endpoint"), store.atomic() as tx:
            tx.append(
                "control",
                lease=lease,
                expected_seq=rows[-1].seq,
                expected_inbox_seq=watermark,
                commit_id="tamper",
                records=(started,),
            )
        assert store.read_state("control")[1] == rows
    finally:
        store.release(lease)


def test_delegation_events_and_child_replay_paired_producer(store, database, tmp_path):
    from vv_agent.event_store import RunEventReplayQuery
    from vv_agent.session.children import ChildSession, child_delivery
    from vv_agent.session.records import SessionSpec

    child = Agent("child", "child")
    from vv_agent.types import SubAgentConfig

    parent = Agent("parent", "parent", sub_agents={"child": SubAgentConfig(model="m", description="child")})
    call = ToolCall("child", "create_sub_task", {"agent_id": "child", "task_description": "child"})
    config = RunConfig(workspace=tmp_path)
    old = old_run(parent, config, [LLMResponse("", [call]), LLMResponse("child done"), LLMResponse("done")])

    agent = parent
    admit(store, config)
    children = {"create_sub_task": lambda _: ChildSession(SessionSpec("child", "test", str(tmp_path)), "child")}
    rt = runtime(store, database, agent, config, ScriptedLLM([LLMResponse("", [call]), LLMResponse("done")]), children=children)
    drive(store, "control", runtime=rt)
    drive(store, "child", runtime=runtime(store, database, child, config, ScriptedLLM([LLMResponse("child done")])))
    with store.atomic() as tx:
        child_delivery(store, tx, "child")
    store._fold_cache = None
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert new.final_output == old.final_output
    for kind in ("sub_run_started", "sub_run_completed"):
        actual = [e for e in new.events if e.type == kind]
        expected = [e for e in old.events if e.type == kind]
        assert len(actual) == len(expected) == 1
        assert actual[0].to_dict()["parent_tool_call_id"] == expected[0].to_dict()["parent_tool_call_id"]
    assert next(e for e in new.events if e.type == "sub_run_completed").to_dict()["status"] == "completed"
    bridge = SessionRunEventStore(store, "control")
    replayed = list(bridge.replay(RunEventReplayQuery(new.run_id, include_children=True)))
    descendants = [e for e in replayed if e.run_id != new.run_id]
    assert descendants and all(e.parent_run_id == new.run_id for e in descendants)
    assert all(event_from_dict(e.to_dict(), _kernel=True).to_dict() == e.to_dict() for e in replayed)


def test_trace_ack_boundary_and_processor_failure(store, database, tmp_path):
    from vv_agent.session.records import InboxItem, SessionSpec

    with store.atomic() as tx:
        tx.create(SessionSpec("control", "test", str(tmp_path)), consumers=("traces",))
        tx.push("control", InboxItem("initial", "user", {"content": "go"}))
    agent, config = Agent("test", "test"), RunConfig(workspace=tmp_path)
    drive(store, "control", runtime=runtime(store, database, agent, config, ScriptedLLM([LLMResponse("done")])))
    seen = []

    class Processor:
        def on_span_start(self, span):
            seen.append(span.span_id)
            raise RuntimeError("telemetry unavailable")

        def on_span_end(self, span):
            seen.append(span.span_id)

        def flush(self):
            raise RuntimeError("telemetry unavailable")

    with store.atomic(), pytest.raises(ValueError, match="top-level transaction"):
        deliver_spans(store, "control", [Processor()])
    assert not seen
    assert deliver_spans(store, "control", [Processor()]) == 4
    saved = list(seen)
    store._fold_cache = None
    assert deliver_spans(store, "control", [Processor()]) == 0 and seen == saved


@pytest.mark.parametrize("output_type", [dict, list, Answer])
def test_output_coercion_exception_becomes_durable_result(store, database, tmp_path, output_type):
    agent, config = Agent("test", "test", output_type=output_type), RunConfig(workspace=tmp_path)
    with pytest.raises(ValueError, match="failed to validate final output"):
        old_run(agent, config, [LLMResponse("invalid")])
    new = new_run(store, database, agent, config, [LLMResponse("invalid")], cut=cut_boundary("output_checked"))
    assert new.status == AgentStatus.FAILED and new.raw_result.error_code == "output_type_invalid"
    assert new.partial_output == "invalid"
    assert len(records(store, kind="op_started")) == 1


def test_uncertain_repair_does_not_retry_on_recovery(store, database, tmp_path):
    calls = []

    def repair(request):
        calls.append(request)
        return "valid"

    agent = Agent(
        "test",
        "test",
        output_validation_enabled=True,
        output_validator=lambda value, _: (
            OutputValidationResult.accept() if value == "valid" else OutputValidationResult.reject("invalid")
        ),
        output_repair=repair,
    )
    config = RunConfig(workspace=tmp_path)
    old = old_run(agent, config, [LLMResponse("invalid")])
    assert old.final_output == "valid" and len(calls) == 1
    calls.clear()

    def cut(point, record):
        if point == "before_external_call" and record.payload["purpose"] == "output_repair":
            raise Restart

    new = new_run(store, database, agent, config, [LLMResponse("invalid")], cut=cut)
    assert not calls and new.status == AgentStatus.FAILED
    repair_calls = [c for c in new.metadata["session_model_calls"] if c["purpose"] == "output_repair"]
    assert len(repair_calls) == 1 and repair_calls[0]["status"] == "ambiguous"
    assert new.raw_result.error_code == "model_outcome_unknown"


@pytest.mark.parametrize("keep", [1, 2, 5])
def test_prompt_too_long_logical_cycle_and_tail_parity(store, database, tmp_path, monkeypatch, keep):
    memory = paired_memory_manager(monkeypatch, 999999, keep)
    config = replace(memory_config(tmp_path), metadata={"memory_keep_recent_messages": keep})
    cycles = []

    class Hook(BaseRuntimeHook):
        def before_memory_compact(self, event):
            cycles.append(event.cycle_index)
            return event.messages

    def too_long(_request):
        raise RuntimeError("maximum context length exceeded")

    agent = Agent("test", "test", hooks=[Hook()])
    steps = [too_long, LLMResponse(SUMMARY), LLMResponse("done")]
    old = old_run(agent, config, steps if keep <= 2 else [too_long, LLMResponse("done")])
    old_cycles = list(cycles)
    cycles.clear()
    new = new_run(store, database, agent, config, steps if keep <= 2 else [too_long, LLMResponse("done")], memory=memory)
    same_result(new, old)
    assert cycles == old_cycles == [1]
    assert [c.index for c in new.raw_cycles] == [c.index for c in old.raw_cycles] == [1]
    assert [(m.role, m.content) for m in new.raw_result.messages] == [(m.role, m.content) for m in old.raw_result.messages]


def test_completed_result_compaction_flags_are_per_cycle(store, database, tmp_path, monkeypatch):
    memory = paired_memory_manager(monkeypatch)
    agent, config = Agent("test", "test", tools=[echo]), memory_config(tmp_path)
    steps = [
        LLMResponse(SUMMARY, raw={"usage": USAGE}),
        LLMResponse("working", [ToolCall("a", "echo", {"text": "one"})], raw={"usage": USAGE}),
        LLMResponse("done", raw={"usage": USAGE}),
    ]
    old = old_run(agent, config, steps)
    new = new_run(store, database, agent, config, steps, memory=memory)
    same_result(new, old)
    assert [c.memory_compacted for c in new.raw_cycles] == [c.memory_compacted for c in old.raw_cycles] == [True, False]
    assert [c.to_dict() for c in new.raw_cycles] == [c.to_dict() for c in old.raw_cycles]


def test_event_observers_cannot_mutate_durable_receipts(store, database, tmp_path):
    agent, config = Agent("test", "test", tools=[echo]), RunConfig(workspace=tmp_path)
    steps = [LLMResponse("working", [ToolCall("a", "echo", {"text": "one", "extra": {"values": [1]}})]), LLMResponse("done")]
    result = new_run(store, database, agent, config, steps)
    bridge = SessionRunEventStore(store, "control")
    original = [e.to_dict() for e in bridge.replay(run_id=result.run_id)]
    for event in bridge.replay(run_id=result.run_id):
        if isinstance(event, ToolCallPlannedEvent | ToolCallStartedEvent):
            event.arguments["extra"]["values"].clear()
        elif isinstance(event, DiagnosticEvent) and event.code == "cycle_llm_response" and event.details["tool_calls"]:
            event.details["tool_calls"][0]["arguments"].clear()
    assert [e.to_dict() for e in bridge.replay(run_id=result.run_id)] == original
    store._fold_cache = None
    assert [e.to_dict() for e in bridge.replay(run_id=result.run_id)] == original


def test_trace_processors_cannot_mutate_durable_output(store, database, tmp_path):
    from vv_agent.session.records import InboxItem, SessionSpec

    with store.atomic() as tx:
        tx.create(SessionSpec("control", "test", str(tmp_path)), consumers=("traces",))
        tx.push("control", InboxItem("initial", "user", {"content": "go"}))
    agent, config = Agent("test", "test", output_type=dict), RunConfig(workspace=tmp_path)
    rt = runtime(store, database, agent, config, ScriptedLLM([LLMResponse('{"items":[1]}')]))
    drive(store, "control", runtime=rt)

    class Processor:
        def on_span_start(self, span):
            pass

        def on_span_end(self, span):
            value = span.metadata.get("final_output")
            if isinstance(value, dict):
                value["items"].clear()

    assert deliver_spans(store, "control", [Processor()]) == 4
    assert project_result(store, "control", "control/turn/initial", runtime=rt).final_output == {"items": [1]}
    store._fold_cache = None
    assert project_result(store, "control", "control/turn/initial", runtime=rt).final_output == {"items": [1]}


def test_span_ids_are_scoped_by_session():
    from vv_agent.session.records import digest
    from vv_agent.session.store import StoredRecord
    from vv_agent.session.tracing import project_spans

    from .helpers import record

    definition = {"agent_name": "test", "task": {}}
    events = [
        project_spans(
            (
                StoredRecord(
                    record("turn_started", sid=sid, definition=definition, definition_digest=digest(definition)), 1, "c", 1, 0
                ),
            )
        )
        for sid in ("one", "two")
    ]
    assert {span.span_id for _, _, span in events[0]}.isdisjoint(span.span_id for _, _, span in events[1])


@pytest.mark.parametrize("policy", list(UnavailableMetricPolicy))
def test_repair_missing_usage_policy_is_durable(store, database, tmp_path, policy):
    agent = Agent(
        "test",
        "test",
        output_validation_enabled=True,
        output_validator=lambda value, _: (
            OutputValidationResult.accept() if value == "valid" else OutputValidationResult.reject("invalid")
        ),
        output_repair=lambda _: "valid",
    )
    config = RunConfig(workspace=tmp_path, budget_limits=RunBudgetLimits(max_total_tokens=100, unavailable_metric_policy=policy))
    old = old_run(agent, config, [LLMResponse("invalid", raw={"usage": USAGE})])
    new = new_run(store, database, agent, config, [LLMResponse("invalid", raw={"usage": USAGE})])
    assert old.final_output == "valid" and old.budget_usage.total_tokens == 15
    assert new.budget_usage.total_tokens is None and new.budget_usage.cycles == 1
    assert new.status == (AgentStatus.FAILED if policy == UnavailableMetricPolicy.STOP else AgentStatus.COMPLETED)
    assert new.budget_exhaustion is not None if policy == UnavailableMetricPolicy.STOP else new.budget_exhaustion is None
    assert new.metadata["session_model_calls"][-1]["purpose"] == "output_repair"
    store._fold_cache = None
    rt = runtime(store, database, agent, config, ScriptedLLM([]))
    rebuilt = project_result(store, "control", new.run_id, runtime=rt)
    assert rebuilt.budget_usage == new.budget_usage and rebuilt.budget_exhaustion == new.budget_exhaustion
