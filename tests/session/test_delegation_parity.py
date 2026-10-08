"""F2d-3 SDK producers versus durable child admission and reconstruction."""

import json
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from threading import Event, Thread
from typing import Any

import pytest
from support import FixedModelProvider, ModelMapProvider
from test_workspace_backends import _FakeS3Client, _make_s3_backend

from vv_agent import Agent, RunConfig, Runner, handoff
from vv_agent.budget import RunBudgetLimits
from vv_agent.canonical_json import canonical_json_bytes
from vv_agent.event_store import RunEventReplayQuery
from vv_agent.events import HandoffCompletedEvent, HandoffStartedEvent
from vv_agent.guardrails import GuardrailResult
from vv_agent.llm.scripted import ScriptedLLM
from vv_agent.run_config import ToolPolicy
from vv_agent.runtime.cancellation import CancelledError
from vv_agent.runtime.sub_task_manager import SubTaskManager
from vv_agent.session.bindings import MissingHostBinding
from vv_agent.session.children import child_delivery, child_handles, completion_input
from vv_agent.session.events import SessionRunEventStore
from vv_agent.session.kernel import drive, read_state
from vv_agent.session.records import InboxItem, Record, RecordError, digest
from vv_agent.session.result import project_result
from vv_agent.session.store import Conflict
from vv_agent.tools.base import ToolContext
from vv_agent.tools.function import function_tool
from vv_agent.types import (
    AgentStatus,
    CompletionReason,
    LLMResponse,
    SubAgentConfig,
    ToolCall,
    ToolDirective,
    ToolExecutionResult,
)

from .conftest import open_store
from .test_runner_parity import RESOLVED
from .test_tools_control_parity import Restart, admit, runtime


def normalize(value):
    if isinstance(value, dict):
        return {
            key: normalize(item)
            for key, item in value.items()
            if key
            not in {
                "task_id",
                "session_id",
                "child_run_id",
                "parent_run_id",
                "parent_tool_call_id",
                "updated_at",
                "recent_activity",
                "task_ids",
            }
        }
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def tool_values(result):
    return [
        (r.status_code, r.error_code, normalize(json.loads(r.content)) if r.content.startswith("{") else r.content)
        for cycle in result.raw_result.cycles
        for r in cycle.tool_results
    ]


def configured():
    return Agent(
        "parent", "Parent instructions.", sub_agents={"worker": SubAgentConfig(model="m", description="Worker instructions.")}
    )


def llm_for(config):
    assert isinstance(config.model_provider, FixedModelProvider)
    return config.model_provider.llm


def provider(steps):
    return FixedModelProvider(ScriptedLLM(deepcopy(steps)), RESOLVED)


def routed(parent_steps, child_steps):
    return ModelMapProvider(
        {
            name: (ScriptedLLM(list(steps)), replace(RESOLVED, requested_model=name, selected_model=name, model_id=name))
            for name, steps in (("parent", parent_steps), ("m", child_steps))
        },
        default_model="parent",
    )


def parent_llm(config):
    assert isinstance(config.model_provider, ModelMapProvider)
    return config.model_provider.routes["parent"][0]


def drain(store, sid, rt):
    # Host scheduler: every drive releases its own lease before descendants run.
    drive(store, sid, runtime=rt)
    for stored in store.read_state(sid)[1]:
        r = stored.record
        if r.kind != "op_parked" or r._payload["handle"]["kind"] != "child":
            continue
        for h in child_handles(r._payload["handle"]):
            child_rt = rt.child_runtime(store, h["session_id"])
            drain(store, h["session_id"], child_rt)
            with store.atomic() as tx:
                child_delivery(store, tx, h["session_id"])
    drive(store, sid, runtime=rt)


def paired(store, database, agent, config, steps, *, cut=None):
    old_config = replace(config, model_provider=provider(steps))
    old = Runner.run_sync(agent, "go", run_config=old_config)
    new_config = replace(config, model_provider=provider(steps))
    admit(store, new_config)
    rt = runtime(store, database, agent, new_config, llm_for(new_config), hook=cut or (lambda *_: None))
    if cut:
        with pytest.raises(Restart):
            drive(store, "control", runtime=rt)
        store._fold_cache = None
        rt = runtime(store, database, agent, new_config, llm_for(new_config))
    drain(store, "control", rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert new.final_output == old.final_output
    assert new.status == old.status
    assert tool_values(new) == tool_values(old)
    return old, new, rt


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("cut_at", ["before", "after", None])
def test_configured_child_atomic_sdk_restart_parity(store, database, tmp_path, batch, cut_at):
    args: dict[str, Any] = {"agent_id": "worker"}
    args.update(
        {"tasks": [{"task_description": "first"}, {"task_description": "second"}]} if batch else {"task_description": "first"}
    )
    steps = [LLMResponse("", [ToolCall("delegate", "create_sub_task", args)])]
    steps += [LLMResponse("child done")] * (2 if batch else 1) + [LLMResponse("parent done")]

    def cut(point, record):
        if point == f"{cut_at}_commit" and record.kind == "op_parked":
            raise Restart

    old, new, rt = paired(store, database, configured(), RunConfig(workspace=tmp_path), steps, cut=cut if cut_at else None)
    parks = [r.record for r in read_state(store, "control")[1] if r.record.kind == "op_parked"]
    assert len(parks) == 1
    handles = child_handles(parks[0].payload["handle"])
    assert len(handles) == (2 if batch else 1)
    rows = read_state(store, "control")[1]
    started = next(r for r in rows if r.record.kind == "op_started" and r.record.operation_id == parks[0].operation_id)
    assert started.commit_id == next(r.commit_id for r in rows if r.record == parks[0])
    for h in handles:
        child_rows = read_state(store, h["session_id"])[1]
        terminal = next(r for r in child_rows if r.record.kind == "turn_ended")
        item = completion_input(h, terminal)
        with store.atomic() as tx:
            assert tx.push("control", item).replayed
        with pytest.raises(Conflict), store.atomic() as tx:
            tx.push("control", replace(item, payload=item.payload | {"result": "forged"}))
    drive(store, "control", runtime=rt)
    assert len([r for r in read_state(store, "control")[1] if r.record.kind == "turn_ended"]) == 1
    assert len([e for e in new.events if e.type == "sub_run_completed"]) == len(
        [e for e in old.events if e.type == "sub_run_completed"]
    )
    keys = {(parks[0].operation_id, parks[0].attempt, h["session_id"]) for h in handles}
    snapshot = store.read_state("control")[0]
    assert snapshot.child_completions is not None
    assert set(snapshot.child_completions) == keys
    snapshot.child_completions.clear()
    assert set(store.read_state("control")[0].child_completions) == keys
    store._fold_cache = None
    assert set(store.read_state("control")[0].child_completions) == keys


@pytest.mark.parametrize("mode", ["as_tool", "configured"])
def test_child_hooks_policy_budget_workspace_inheritance_frozen(store, database, tmp_path, mode):
    seen = []

    @function_tool
    def inspect(ctx: ToolContext) -> str:
        seen.append((ctx.workspace, ctx.workspace_backend, ctx.task_metadata, ctx.shared_state.get("seed")))
        return "inspected"

    @function_tool
    def forbidden() -> str:
        raise AssertionError("inherited denial must win")

    child = Agent("worker", "Worker instructions.", tools=[inspect, forbidden])
    parent = configured() if mode == "configured" else Agent("parent", "Parent instructions.", tools=[child.as_tool()])
    if mode == "configured":
        parent.tools = [inspect, forbidden]
        parent.sub_agents["worker"].exclude_tools = ["forbidden"]
        parent.sub_agents["worker"].system_prompt = "Frozen configured instructions."
        parent.metadata = {
            "bash_shell": "bash",
            "windows_shell_priority": ["powershell", "cmd"],
            "bash_env": {"CHILD_VALUE": "汉字 😀"},
            "allow_outside_workspace_paths": False,
            "language": "en-US",
            "available_skills": [],
            "active_skills": [],
        }
    name = "create_sub_task" if mode == "configured" else "worker"
    args = {"task_description": "work", **({"agent_id": "worker"} if mode == "configured" else {})}
    args.update({"output_requirements": "Required output", "include_main_summary": True})
    prompts = []

    def child_call(request):
        prompts.append(next(m.content for m in request.messages if m.role == "user"))
        return LLMResponse("", [ToolCall("inspect", "inspect", {})])

    steps = [
        LLMResponse("", [ToolCall("delegate", name, args)]),
        child_call,
        LLMResponse("child done"),
        LLMResponse("parent done"),
    ]
    config = RunConfig(
        workspace=tmp_path,
        tool_policy=ToolPolicy(disallowed_tools=["forbidden"]),
        budget_limits=RunBudgetLimits(max_tool_calls=4),
        shared_state={"seed": "parent"},
    )
    old, new, rt = paired(store, database, parent, config, steps)
    assert len(seen) == 2 and seen[0][0] == seen[1][0] == tmp_path
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    assert "Required output" in prompts[0] and "Main Task Summary" in prompts[0]
    child_id = child_handles(
        next(r.record.payload["handle"] for r in store.read_state("control")[1] if r.record.kind == "op_parked")
    )[0]["session_id"]
    admission = store.read(child_id, limit=1).records[0].record.payload["attributes"]["child_admission"]
    assert admission["budget"]["max_tool_calls"] == 4
    assert "forbidden" in admission["definition"]["task"]["metadata"]["_vv_agent_disallowed_tools"]
    if mode == "configured":
        assert seen[0][3] is seen[1][3] is None
        for key, value in parent.metadata.items():
            assert seen[0][2][key] == seen[1][2][key] == value
        parent.sub_agents["worker"].system_prompt = "Changed after admission."
        parent.sub_agents["worker"].max_cycles = 1
        parent.metadata["bash_env"]["CHILD_VALUE"] = "changed"
        rebuilt = rt.child_runtime(store, child_id)
        assert rebuilt.frozen_task.prompt_bundle.flatten() == "Frozen configured instructions."
        assert rebuilt.frozen_task.max_cycles == 8
        assert rebuilt.frozen_task.metadata["bash_env"] == {"CHILD_VALUE": "汉字 😀"}
    else:
        assert seen[0][3] == seen[1][3] == "parent"
    assert new.budget_usage.tool_calls == old.budget_usage.tool_calls


@pytest.mark.parametrize("mode", ["configured", "as_tool"])
def test_sdk_children_never_recursive_runner_or_parent_lease(store, database, tmp_path, monkeypatch, mode):
    child = Agent("worker", "Worker instructions.")
    agent = configured() if mode == "configured" else Agent("parent", "Parent instructions.", tools=[child.as_tool()])
    call = ToolCall(
        "delegate",
        "create_sub_task" if mode == "configured" else "worker",
        {"task_description": "work", **({"agent_id": "worker"} if mode == "configured" else {})},
    )
    steps = [LLMResponse("", [call]), LLMResponse("child done"), LLMResponse("parent done")]
    old = Runner.run_sync(agent, "go", run_config=RunConfig(workspace=tmp_path, model_provider=provider(steps)))
    config = RunConfig(workspace=tmp_path, model_provider=provider(steps))

    def forbidden(*args, **kwargs):
        raise AssertionError("kernel must not call Runner")

    monkeypatch.setattr(Runner, "run_sync", forbidden)
    monkeypatch.setattr(Runner, "start", forbidden)
    admit(store, config)
    rt = runtime(store, database, agent, config, llm_for(config))
    drive(store, "control", runtime=rt)
    assert read_state(store, "control")[0].phase == "parked"
    lease = store.acquire("control", owner="check-parent-released", ttl_ms=15000)
    assert lease is not None
    store.release(lease)
    drain(store, "control", rt)
    assert project_result(store, "control", "control/turn/initial", runtime=rt).final_output == old.final_output


@pytest.mark.parametrize(
    "invalid",
    [
        {"agent_id": "unknown", "task_description": "work"},
        {"agent_id": "worker", "tasks": []},
        {"agent_id": "worker", "task_description": "work", "tasks": [{}]},
        {"agent_id": "worker", "task_description": "work", "wait_for_completion": 1},
        {"agent_id": "worker", "task_description": "work", "exclude_files_pattern": "["},
    ],
)
def test_configured_child_argument_failure_parity(store, database, tmp_path, invalid):
    paired(
        store,
        database,
        configured(),
        RunConfig(workspace=tmp_path),
        [LLMResponse("", [ToolCall("delegate", "create_sub_task", invalid)]), LLMResponse("done")],
    )
    assert store.list_sessions() == ("control",)


def test_sub_task_status_records_projection_parity(store, database, tmp_path):
    def status(request):
        child = json.loads(next(m.content for m in request.messages if m.role == "tool"))
        return LLMResponse(
            "", [ToolCall("status", "sub_task_status", {"task_ids": [child["task_id"]], "detail_level": "snapshot"})]
        )

    _old, _new, rt = paired(
        store,
        database,
        configured(),
        RunConfig(workspace=tmp_path),
        [
            LLMResponse("", [ToolCall("delegate", "create_sub_task", {"agent_id": "worker", "task_description": "work"})]),
            LLMResponse("child done"),
            status,
            LLMResponse("done"),
        ],
    )
    child_id = next(s for s in store.list_sessions() if s != "control")
    store._fold_cache = None
    rebuilt = runtime(store, database, rt.agent, rt.config, rt.llm)
    assert rebuilt.child_tasks(store, "control").get(child_id).outcome.final_answer == "child done"
    assert rebuilt.child_tasks(store, "another-parent").get(child_id) is None
    assert rebuilt.child_tasks(store, "control").get("missing") is None


@pytest.mark.parametrize("cancel", [False, True])
def test_background_handle_start_poll_wait_cancel_reconstruction(store, database, tmp_path, cancel):
    entered, release = Event(), Event()

    def blocking(_request):
        entered.set()
        assert release.wait(10)
        return LLMResponse("child done")

    child = Agent("worker", "Work.")
    tool = child.as_background_task()
    config = RunConfig(workspace=tmp_path, model_provider=provider([blocking]))
    old_handle = tool.start(Runner, None, {"task_description": "work"}, run_config=config)
    assert entered.wait(5)
    assert old_handle.poll().status == AgentStatus.RUNNING
    with pytest.raises(TimeoutError):
        old_handle.wait(0)
    if cancel:
        old_handle._run_handle.cancel()
    release.set()
    old = old_handle.wait(5)
    steps = [LLMResponse("", [ToolCall("background", tool.name, {"task_description": "work"})]), LLMResponse("parent done")]
    config = RunConfig(workspace=tmp_path, model_provider=provider(steps))
    parent = Agent("parent", "Parent.", tools=[tool])
    admit(store, config)
    rt = runtime(store, database, parent, config, llm_for(config))
    drive(store, "control", runtime=rt)
    child_id = next(s for s in store.list_sessions() if s != "control")
    store._fold_cache = None
    rt = runtime(store, database, parent, config, llm_for(config))
    handle = rt.child_tasks(store, "control").handle(child_id)
    assert handle.poll().status == AgentStatus.RUNNING
    with pytest.raises(TimeoutError):
        handle.wait(0)
    if cancel:
        handle.cancel()
        handle.cancel()
        assert len([i for i in store.peek_inbox(child_id) if i.item.kind == "control"]) == 1
    child_rt = rt.child_runtime(store, child_id)
    child_rt.llm = ScriptedLLM([] if cancel else [LLMResponse("child done")])
    drive(store, child_id, runtime=child_rt)
    actual = handle.wait(0)
    assert actual.status == old.status
    assert actual.final_output == old.final_output
    assert actual.done and handle.status == actual.status
    with pytest.raises(KeyError):
        rt.child_tasks(store, "another-parent").handle(child_id)
    with store.atomic() as tx:
        child_delivery(store, tx, child_id)
    drive(store, "control", runtime=rt)
    assert len([r for r in store.read_state("control")[1] if r.record.kind == "turn_ended"]) == 1


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_handoff_durable_transfer_and_maximum_parity(store, database, tmp_path, depth):
    third = Agent("third", "Third.")
    second = Agent("second", "Second.", handoffs=[handoff(agent=third)])
    first = Agent("first", "First.", handoffs=[handoff(agent=second)])
    steps = [
        LLMResponse("", [ToolCall("first-transfer", "transfer_to_second", {"input": "summary"})]),
        LLMResponse("", [ToolCall("second-transfer", "transfer_to_third", {"input": "next summary"})]),
        LLMResponse("transferred done"),
    ]
    config = RunConfig(workspace=tmp_path, max_handoffs=depth, model_provider=provider(steps))
    if depth < 2:
        with pytest.raises(RuntimeError, match="maximum handoff depth exceeded"):
            Runner.run_sync(first, "go", run_config=config)
    else:
        old = Runner.run_sync(first, "go", run_config=config)
    config = replace(config, model_provider=provider(steps))
    admit(store, config)
    rt = runtime(store, database, first, config, llm_for(config))

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_parked":
            raise Restart

    if depth:
        rt.hook = cut
        with pytest.raises(Restart):
            drive(store, "control", runtime=rt)
        rt = runtime(store, database, first, replace(config, max_handoffs=depth + 100), llm_for(config))
        store._fold_cache = None
    drain(store, "control", rt)
    rows = [store.read(s, limit=1).records[0].record for s in store.list_sessions() if s != "control"]
    assert sorted(r.payload["attributes"]["child_admission"]["handoff_count"] for r in rows) == list(range(1, depth + 1))
    result = project_result(store, "control", "control/turn/initial", runtime=rt)
    if depth == 2:
        assert result.final_output == old.final_output == "transferred done"
        assert result.agent_name == old.agent_name == "third"
    else:
        assert result.status == AgentStatus.FAILED
        assert result.raw_result.error_code == "maximum_handoffs_exceeded"
    before = {s: store.read(s).head_seq for s in store.list_sessions()}
    drive(store, "control", runtime=rt)
    assert before == {s: store.read(s).head_seq for s in store.list_sessions()}


def test_shared_state_host_binding_restart_intentional_difference(store, database, tmp_path):
    class Host:
        def __deepcopy__(self, memo):
            return self

    original = Host()
    seen = []

    @function_tool
    def bound(ctx: ToolContext) -> str:
        seen.append(ctx.shared_state["host"])
        ctx.shared_state["count"] += 1
        return str(ctx.shared_state["count"])

    agent = Agent("bound", "Bound.", tools=[bound])
    steps = [LLMResponse("", [ToolCall("a", "bound", {}), ToolCall("b", "bound", {})]), LLMResponse("done")]
    old = Runner.run_sync(
        agent,
        "go",
        run_config=RunConfig(workspace=tmp_path, shared_state={"host": original, "count": 0}, model_provider=provider(steps)),
    )
    assert seen == [original, original]
    config = RunConfig(workspace=tmp_path, shared_state={"count": 0}, model_provider=provider(steps))
    admit(store, config)

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_completed" and record.operation_id.endswith("/tool/0"):
            raise Restart

    rt = runtime(store, database, agent, config, llm_for(config), host_bindings={"host": original}, hook=cut)
    with pytest.raises(Restart):
        drive(store, "control", runtime=rt)
    rows = store.read_state("control")[1]
    assert all('"host":' not in r.record.encode().decode() for r in rows)
    store._fold_cache = None
    missing = runtime(store, database, agent, config, llm_for(config))
    with pytest.raises(MissingHostBinding, match="host"):
        drive(store, "control", runtime=missing)
    assert store.read_state("control")[1] == rows
    supplied = Host()
    rt = runtime(store, database, agent, config, llm_for(config), host_bindings={"host": supplied})
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert new.final_output == old.final_output
    assert new.raw_result.shared_state["count"] == old.raw_result.shared_state["count"] == 2
    assert seen == [original, original, original, supplied]


@pytest.mark.parametrize("change", ["replace", "delete", "arbitrary", "collision"])
def test_host_binding_json_boundary_rejects_invalid_state(store, database, tmp_path, change):
    obj = object()

    @function_tool
    def mutate(ctx: ToolContext) -> str:
        if change == "replace":
            ctx.shared_state["host"] = object()
        elif change == "delete":
            del ctx.shared_state["host"]
        else:
            ctx.shared_state["other"] = object()
        return "mutated"

    config = RunConfig(workspace=tmp_path, shared_state={"host": 1} if change == "collision" else {})
    admit(store, config)
    rt = runtime(
        store,
        database,
        Agent("bound", "Bound.", tools=[mutate]),
        config,
        ScriptedLLM([LLMResponse("", [ToolCall("mutate", "mutate", {})])]),
        host_bindings={"host": obj},
    )
    if change == "collision":
        with pytest.raises(ValueError, match="shadow"):
            drive(store, "control", runtime=rt)
    else:
        with pytest.raises((ValueError, TypeError)):
            drive(store, "control", runtime=rt)
    assert not any(
        r.record.kind == "op_completed" and "/tool/" in (r.record.operation_id or "") for r in store.read_state("control")[1]
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_s3_streaming_workspace_paired_producers(store, database, tmp_path, streaming):
    bodies = []

    class Body(BytesIO):
        source: bool = False

        def read(self, size=-1, /):
            if streaming and getattr(self, "source", False):
                assert 0 < size <= 65536
            return super().read(size)

    class Client(_FakeS3Client):
        def get_object(self, **kwargs):
            body = Body(self.objects[kwargs["Key"]])
            body.source = kwargs["Key"].endswith("/data.txt")
            bodies.append(body)
            return {"Body": body}

    def setup():
        backend = _make_s3_backend()
        client = Client()
        client.objects = backend._client.objects
        backend._client = client
        backend.write_text("data.txt", "中\n" * 80000)
        return backend

    def model(request):
        assert any(m.role == "tool" and "中" in m.content for m in request.messages)
        return LLMResponse("done")

    steps = [
        LLMResponse("", [ToolCall("read", "read_file", {"path": "data.txt", "end_line": 2})]),
        LLMResponse("", [ToolCall("write", "write_file", {"path": "output.txt", "content": "replacement"})]),
        model,
    ]
    old_backend = setup()
    old = Runner.run_sync(
        Agent("s3", "S3."),
        "go",
        run_config=RunConfig(workspace=tmp_path, workspace_backend=old_backend, model_provider=provider(steps)),
    )
    new_backend = setup()
    config = RunConfig(workspace=tmp_path, workspace_backend=new_backend, model_provider=provider(steps))
    admit(store, config)

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_completed" and record.operation_id.endswith("/tool/0"):
            raise Restart

    rt = runtime(store, database, Agent("s3", "S3."), config, llm_for(config), hook=cut)
    with pytest.raises(Restart):
        drive(store, "control", runtime=rt)
    store._fold_cache = None
    rt = runtime(store, database, Agent("s3", "S3."), config, llm_for(config))
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert new.final_output == old.final_output
    assert tool_values(new) == tool_values(old)
    assert new_backend._client.objects[f"{new_backend._prefix}/output.txt"] == b"replacement"
    assert new_backend._client.objects == old_backend._client.objects
    assert bodies and all(b.closed for b in bodies if b.source)


@pytest.mark.parametrize("mode", ["configured", "batch", "async", "as_tool", "background"])
def test_sdk_parent_cancellation_cascades_after_reconstruction(store, database, tmp_path, mode):
    entered, release = Event(), Event()

    def block(_request):
        entered.set()
        assert release.wait(10)
        return LLMResponse("cancelled child")

    child = Agent("worker", "Work.", model="m")
    tool = child.as_background_task() if mode == "background" else child.as_tool()
    parent = configured() if mode in {"configured", "batch", "async"} else Agent("parent", "Parent.", tools=[tool])
    parent.model = "parent"
    args: dict[str, Any] = {"task_description": "work"}
    if parent.sub_agents:
        args["agent_id"] = "worker"
    if mode == "batch":
        args = {"agent_id": "worker", "tasks": [{"task_description": "one"}, {"task_description": "two"}]}
    if mode == "async":
        args["wait_for_completion"] = False
    call = LLMResponse("", [ToolCall("delegate", "create_sub_task" if parent.sub_agents else tool.name, args)])
    manager = SubTaskManager(register_session=lambda *_: None, unregister_session=lambda *_: None)
    old_provider = routed([call, block], [block] * (2 if mode == "batch" else 1))
    old_handle = Runner.start(
        parent, "go", run_config=RunConfig(workspace=tmp_path, model_provider=old_provider, sub_task_manager=manager)
    )
    try:
        assert entered.wait(5)
        old_handle.cancel()
    finally:
        release.set()
    if mode == "as_tool":
        with pytest.raises(CancelledError):
            old_handle.result(5)
        old = None
    else:
        old = old_handle.result(5)
        assert old.completion_reason == CompletionReason.CANCELLED
    for record in list(manager._tasks.values()):
        done = manager.wait(record.task_id, 5)
        assert done is not None and done.outcome is not None
        assert done.outcome.status == AgentStatus.FAILED
        assert done.outcome.completion_reason == CompletionReason.CANCELLED

    config = RunConfig(workspace=tmp_path, model_provider=routed([call], []))
    admit(store, config)

    def cut(point, record):
        if point == "after_commit" and (
            record.kind == "op_parked"
            or (mode in {"async", "background"} and record.kind == "op_completed" and "/tool/" in record.operation_id)
        ):
            raise Restart

    rt = runtime(store, database, parent, config, parent_llm(config), hook=cut)
    with pytest.raises(Restart):
        drive(store, "control", runtime=rt)
    store._fold_cache = None
    rt = runtime(store, database, parent, config, parent_llm(config))
    with store.atomic() as tx:
        tx.push("control", InboxItem("cancel-parent", "control", {"action": "cancel"}, "control/turn/initial"))
    drive(store, "control", runtime=rt)
    ids = [sid for sid in store.list_sessions() if sid != "control"]
    assert len(ids) == (2 if mode == "batch" else 1)
    for sid in ids:
        assert [item.item.payload for item in store.peek_inbox(sid) if item.item.kind == "control"] == [{"action": "cancel"}]
        child_rt = rt.child_runtime(store, sid)
        drive(store, sid, runtime=child_rt)
        snapshot = rt.child_tasks(store, "control").handle(sid).poll()
        assert snapshot.status == AgentStatus.FAILED
        with store.atomic() as tx:
            child_delivery(store, tx, sid)
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert new.status == AgentStatus.FAILED and new.completion_reason == CompletionReason.CANCELLED
    if old is not None:
        assert new.status == old.status and new.completion_reason == old.completion_reason
    assert len([r for r in store.read_state("control")[1] if r.record.kind == "turn_ended"]) == 1
    assert not any(
        r.record.payload["context"] == "normal"
        for r in store.read_state("control")[1]
        if r.record.kind == "op_completed" and r.record.operation_id.endswith("/tool/0") and mode not in {"async", "background"}
    )


@pytest.mark.parametrize("batch", [False, True])
def test_configured_async_admission_status_and_late_delivery_parity(store, database, tmp_path, batch):
    entered, release = Event(), Event()

    def block(_request):
        entered.set()
        assert release.wait(10)
        return LLMResponse("child done")

    def final(_request):
        assert entered.wait(5)
        return LLMResponse("parent done")

    args: dict[str, Any] = {"agent_id": "worker", "wait_for_completion": False}
    args.update({"tasks": [{"task_description": "one"}, {"task_description": "two"}]} if batch else {"task_description": "one"})
    call = LLMResponse("", [ToolCall("delegate", "create_sub_task", args)])
    parent = configured()
    parent.model = "parent"
    manager = SubTaskManager(register_session=lambda *_: None, unregister_session=lambda *_: None)
    old_config = RunConfig(
        workspace=tmp_path, model_provider=routed([call, final], [block] * (2 if batch else 1)), sub_task_manager=manager
    )
    try:
        old = Runner.run_sync(parent, "go", run_config=old_config)
    finally:
        release.set()
    for record in list(manager._tasks.values()):
        done = manager.wait(record.task_id, 5)
        assert done is not None and done.outcome is not None
        assert done.outcome.final_answer == "child done"
    config = RunConfig(
        workspace=tmp_path,
        model_provider=routed([call, LLMResponse("parent done")], [LLMResponse("child done")] * (2 if batch else 1)),
    )
    admit(store, config)
    rt = runtime(store, database, parent, config, parent_llm(config))
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert new.final_output == old.final_output
    assert tool_values(new) == tool_values(old)
    ids = [sid for sid in store.list_sessions() if sid != "control"]
    assert len([e for e in new.events if e.type == "sub_run_started"]) == len(ids)
    for sid in ids:
        assert rt.child_tasks(store, "control").get(sid).is_running()
        drive(store, sid, runtime=rt.child_runtime(store, sid))
        with store.atomic() as tx:
            child_delivery(store, tx, sid)
    drive(store, "control", runtime=rt)
    assert all(rt.child_tasks(store, "control").get(sid).outcome.final_answer == "child done" for sid in ids)
    assert len([r for r in store.read_state("control")[1] if r.record.kind == "turn_ended"]) == 1
    assert all(
        r.record.payload["disposition"] == "noop"
        for r in store.read_state("control")[1]
        if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "child_result"
    )


def test_sub_task_status_message_continuation_and_wait_projection_parity(store, database, tmp_path):
    def status(request):
        payload = json.loads(next(m.content for m in request.messages if m.role == "tool"))
        return LLMResponse(
            "",
            [
                ToolCall(
                    "status",
                    "sub_task_status",
                    {"task_ids": [payload["task_id"]], "message": "continue", "wait_for_response": True},
                )
            ],
        )

    parent = configured()
    parent.model = "parent"
    parent_steps = [
        LLMResponse("", [ToolCall("delegate", "create_sub_task", {"agent_id": "worker", "task_description": "work"})]),
        status,
        LLMResponse("parent done"),
    ]
    child_steps = [LLMResponse("first result"), LLMResponse("second result")]
    old = Runner.run_sync(
        parent, "go", run_config=RunConfig(workspace=tmp_path, model_provider=routed(parent_steps, child_steps))
    )
    config = RunConfig(workspace=tmp_path, model_provider=routed(parent_steps, child_steps))
    admit(store, config)
    rt = runtime(store, database, parent, config, parent_llm(config))
    drive(store, "control", runtime=rt)
    sid = next(s for s in store.list_sessions() if s != "control")
    drive(store, sid, runtime=rt.child_runtime(store, sid))
    with store.atomic() as tx:
        child_delivery(store, tx, sid)
    rt = runtime(store, database, parent, config, rt.llm)
    errors = []

    def parent_worker():
        try:
            with open_store(database) as worker_store:
                drive(worker_store, "control", runtime=rt)
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=parent_worker)
    worker.start()
    try:
        for _ in range(500):
            if any(i.item.kind == "follow_up" for i in store.peek_inbox(sid)):
                break
            Event().wait(0.01)
        else:
            pytest.fail("status message did not arrive")
        assert rt.child_tasks(store, "control").get(sid).is_running()
        store._fold_cache = None
        drive(store, sid, runtime=rt.child_runtime(store, sid))
    finally:
        worker.join(5)
    assert not worker.is_alive() and not errors, errors
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert tool_values(new) == tool_values(old)
    assert rt.child_tasks(store, "control").get(sid).outcome.final_answer == "second result"
    manager = rt.child_tasks(store, "control")
    message = next(
        r.record.payload["input"]
        for r in store.read_state(sid)[1]
        if r.record.kind == "input_applied" and r.record.payload["input"]["kind"] == "follow_up"
    )
    assert manager.message(sid, message["input_id"], "continue") == "continued"
    with pytest.raises(Conflict):
        manager.message(sid, message["input_id"], "different")
    with pytest.raises(KeyError):
        rt.child_tasks(store, "foreign").message(sid, "foreign", "go")
    with store.atomic() as tx:
        child_delivery(store, tx, sid)
    assert not store.peek_inbox("control")
    assert len([r for r in store.read_state(sid)[1] if r.record.kind == "turn_ended"]) == 2


@pytest.mark.parametrize("mode", ["configured", "as_tool"])
def test_blocking_child_user_wait_is_durable_intentional_difference(store, database, tmp_path, mode):
    @function_tool
    def wait_child() -> ToolExecutionResult:
        return ToolExecutionResult("", "Need input", directive=ToolDirective.WAIT_USER, metadata={"question": "Need input"})

    child = Agent("worker", "Work.", tools=[wait_child])
    parent = configured() if mode == "configured" else Agent("parent", "Parent.", tools=[child.as_tool()])
    if mode == "configured":
        parent.tools = [wait_child]
    call = LLMResponse(
        "",
        [
            ToolCall(
                "delegate",
                "create_sub_task" if mode == "configured" else "worker",
                {"task_description": "work", **({"agent_id": "worker"} if mode == "configured" else {})},
            )
        ],
    )
    waiting = LLMResponse("", [ToolCall("wait", "wait_child", {})])
    old = Runner.run_sync(
        parent,
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=provider([call, waiting, LLMResponse("parent done")])),
    )
    assert old.status == AgentStatus.COMPLETED
    old_child_result = old.raw_result.cycles[0].tool_results[0]
    if mode == "configured":
        assert old_child_result.error_code == "sub_task_wait_user"
    else:
        assert old_child_result.metadata["child_status"] == "wait_user"
    config = RunConfig(
        workspace=tmp_path, model_provider=provider([call, waiting, LLMResponse("child done"), LLMResponse("parent done")])
    )
    admit(store, config)
    rt = runtime(store, database, parent, config, llm_for(config))
    drive(store, "control", runtime=rt)
    sid = next(s for s in store.list_sessions() if s != "control")
    drive(store, sid, runtime=rt.child_runtime(store, sid))
    assert rt.child_tasks(store, "control").get(sid).outcome.status == AgentStatus.WAIT_USER
    assert read_state(store, "control")[0].phase == "parked"
    with store.atomic() as tx:
        child_delivery(store, tx, sid)
    assert not store.peek_inbox("control")
    rt = runtime(store, database, parent, config, llm_for(config))
    store._fold_cache = None
    rt.child_tasks(store, "control").message(sid, "reply", "supplied input")
    drive(store, sid, runtime=rt.child_runtime(store, sid))
    with store.atomic() as tx:
        child_delivery(store, tx, sid)
    drive(store, "control", runtime=rt)
    assert project_result(store, "control", "control/turn/initial", runtime=rt).final_output == old.final_output
    assert rt.child_tasks(store, "control").get(sid).outcome.final_answer == "child done"


@pytest.mark.parametrize("blocked", [False, True])
def test_handoff_target_validation_state_and_events_parity(store, database, tmp_path, blocked):
    seen = []

    @function_tool
    def inspect(ctx: ToolContext) -> str:
        seen.append(ctx.shared_state["seed"])
        ctx.shared_state["seed"] = "target"
        return "inspected"

    source_checks = []

    def source_output(_context, value):
        assert json.loads(value)["handoff"] is True
        source_checks.append(value)
        return GuardrailResult.allow()

    def target_output(_context, value):
        seen.append("output")
        return GuardrailResult.rewrite(value + " checked")

    child = Agent(
        "worker",
        "Work.",
        tools=[inspect],
        input_guardrails=[lambda *_: GuardrailResult.block("target blocked")] if blocked else [],
        output_guardrails=[] if blocked else [target_output],
    )
    parent = Agent(
        "parent", "Parent.", handoffs=[handoff(agent=child, metadata={"route": "writing"})], output_guardrails=[source_output]
    )
    steps = [LLMResponse("", [ToolCall("transfer", "transfer_to_worker", {"input": "summary"})])]
    if not blocked:
        steps += [LLMResponse("", [ToolCall("inspect", "inspect", {})]), LLMResponse("target done")]
    config = RunConfig(workspace=tmp_path, shared_state={"seed": "source"})
    old = Runner.run_sync(parent, "go", run_config=replace(config, model_provider=provider(steps)))
    config = replace(config, model_provider=provider(steps))
    admit(store, config)
    rt = runtime(store, database, parent, config, llm_for(config))
    drain(store, "control", rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert new.status == old.status and new.final_output == old.final_output
    assert new.agent_name == old.agent_name == "worker"
    assert len(source_checks) == 1  # Kernel retains a live parent wait instead of validating Runner's transfer marker.
    if not blocked:
        assert new.raw_result.shared_state["seed"] == old.raw_result.shared_state["seed"] == "target"
        assert seen == ["source", "output", "source", "output"]
    events = list(
        SessionRunEventStore(store, "control").replay(RunEventReplayQuery(run_id="control/turn/initial", include_children=True))
    )
    transfers = [e for e in events if isinstance(e, (HandoffStartedEvent, HandoffCompletedEvent))]
    old_transfers = [e for e in old.events if isinstance(e, (HandoffStartedEvent, HandoffCompletedEvent))]
    assert [(e.type, e.source_agent, e.target_agent, e.status, e.metadata["route"]) for e in transfers] == [
        (e.type, e.source_agent, e.target_agent, e.status, e.metadata["route"]) for e in old_transfers
    ]
    assert len({e.event_id for e in events}) == len(events)
    assert [e.to_dict() for e in events] == [
        e.to_dict()
        for e in SessionRunEventStore(store, "control").replay(
            RunEventReplayQuery(run_id="control/turn/initial", include_children=True)
        )
    ]


@pytest.mark.parametrize("mutation", ["extra", "sub_config", "digest", "siblings", "marker"])
def test_child_admission_closed_fields_and_rollback(store, database, tmp_path, mutation):
    config = RunConfig(
        workspace=tmp_path,
        model_provider=provider(
            [
                LLMResponse(
                    "",
                    [
                        ToolCall(
                            "delegate",
                            "create_sub_task",
                            {"agent_id": "worker", "tasks": [{"task_description": "one"}, {"task_description": "two"}]},
                        )
                    ],
                )
            ]
        ),
    )
    admit(store, config)
    rt = runtime(store, database, configured(), config, llm_for(config))
    drive(store, "control", runtime=rt)
    sid = next(s for s in store.list_sessions() if s != "control")
    retained = store.read(sid, limit=1).records[0].record
    assert retained.encode() == canonical_json_bytes(retained.to_dict())
    created = retained.to_dict()
    admission = created["payload"]["attributes"]["child_admission"]
    if mutation in {"extra", "sub_config", "digest"}:
        if mutation == "extra":
            admission["extra"] = True
        elif mutation == "sub_config":
            admission["sub_config"]["extra"] = True
        else:
            admission["definition"]["task"]["user_prompt"] = "tampered"
        value = created
    else:
        value = next(r.record.to_dict() for r in store.read_state("control")[1] if r.record.kind == "op_parked")
        if mutation == "siblings":
            value["payload"]["handle"]["siblings"][0]["extra"] = True
        else:
            value["payload"]["delegation"]["handoff_count"] = True
    before = store.list_sessions()
    with pytest.raises(RecordError), store.atomic() as tx:
        tx.push("control", InboxItem("rolled-back", "user", {"content": "go"}))
        Record.parse(json.dumps(value).encode())
    assert store.list_sessions() == before and not store.peek_inbox("control")
    with pytest.raises(Conflict, match="handler version"):
        replace(rt, handler_version="new").child_runtime(store, sid)
    assert digest(admission["definition"]) == admission["definition_digest"] or mutation == "digest"


@pytest.mark.parametrize("mode", ["configured", "as_tool"])
@pytest.mark.parametrize("budget_stop", [False, True])
def test_child_inherited_denial_and_budget_execute_real_producers(store, database, tmp_path, mode, budget_stop):
    @function_tool
    def forbidden() -> str:
        raise AssertionError("a denied child tool must never execute")

    child = Agent("worker", "Work.", tools=[forbidden])
    parent = configured() if mode == "configured" else Agent("parent", "Parent.", tools=[child.as_tool()])
    if mode == "configured":
        parent.tools = [forbidden]
        parent.sub_agents["worker"].denied_capability_tags = ["forbidden"]
    args = {"task_description": "work", **({"agent_id": "worker"} if mode == "configured" else {})}
    call = LLMResponse("", [ToolCall("delegate", "create_sub_task" if mode == "configured" else "worker", args)])
    steps = [call, LLMResponse("", [ToolCall("denied", "forbidden", {})])]
    if budget_stop:
        steps[1].tool_calls.append(ToolCall("other-denied", "forbidden", {}))
    else:
        steps.append(LLMResponse("child done"))
    steps.append(LLMResponse("parent done"))
    config = RunConfig(
        workspace=tmp_path,
        tool_policy=ToolPolicy(disallowed_tools=["forbidden"]),
        budget_limits=RunBudgetLimits(max_tool_calls=1) if budget_stop else None,
    )
    old, new, rt = paired(store, database, parent, config, steps)
    sid = next(s for s in store.list_sessions() if s != "control")
    outcome = rt.child_tasks(store, "control").get(sid).outcome
    if budget_stop:
        assert outcome.error_code == ("sub_task_failed" if mode == "configured" else "run_budget_exhausted")
    else:
        child_result = project_result(store, sid, f"{sid}/turn/start", runtime=rt.child_runtime(store, sid))
        assert child_result.raw_result.cycles[0].tool_results[0].status_code.value == "ERROR"
    assert new.status == old.status


def test_completion_projection_uses_authenticated_turn_not_later_continuation(store, database, tmp_path, monkeypatch):
    parent = configured()
    parent.model = "parent"
    call = LLMResponse("", [ToolCall("delegate", "create_sub_task", {"agent_id": "worker", "task_description": "work"})])
    old = Runner.run_sync(
        parent,
        "go",
        run_config=RunConfig(
            workspace=tmp_path, model_provider=routed([call, LLMResponse("parent done")], [LLMResponse("initial result")])
        ),
    )
    config = RunConfig(
        workspace=tmp_path,
        model_provider=routed([call, LLMResponse("parent done")], [LLMResponse("initial result"), LLMResponse("later result")]),
    )
    admit(store, config)
    rt = runtime(store, database, parent, config, parent_llm(config))
    drive(store, "control", runtime=rt)
    sid = next(s for s in store.list_sessions() if s != "control")
    drive(store, sid, runtime=rt.child_runtime(store, sid))
    with store.atomic() as tx:
        child_delivery(store, tx, sid)
    rt.child_tasks(store, "control").message(sid, "continue", "continue")
    drive(store, sid, runtime=rt.child_runtime(store, sid))
    assert rt.child_tasks(store, "control").get(sid).outcome.final_answer == "later result"
    store._fold_cache = None
    rt = runtime(store, database, parent, config, parent_llm(config))
    from vv_agent.session import runtime as bindings

    builds = []
    original = bindings.build_default_registry

    def registry():
        builds.append(True)
        return original()

    monkeypatch.setattr(bindings, "build_default_registry", registry)
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert tool_values(new) == tool_values(old)
    assert new.raw_result.cycles[0].tool_results[0].metadata["final_answer"] == "initial result"
    assert len(builds) == 1, "only the executing parent needs tool registration; child result projection is read-only"


@pytest.mark.parametrize("cut", ["before_parent_inbox", "after_parent_inbox", "after_child_ack"])
def test_sdk_completion_delivery_failure_cut_and_replay_parity(store, database, tmp_path, cut):
    steps = [
        LLMResponse("", [ToolCall("delegate", "create_sub_task", {"agent_id": "worker", "task_description": "work"})]),
        LLMResponse("child done"),
        LLMResponse("parent done"),
    ]
    old = Runner.run_sync(configured(), "go", run_config=RunConfig(workspace=tmp_path, model_provider=provider(steps)))
    config = RunConfig(workspace=tmp_path, model_provider=provider(steps))
    admit(store, config)
    rt = runtime(store, database, configured(), config, llm_for(config))
    drive(store, "control", runtime=rt)
    sid = next(s for s in store.list_sessions() if s != "control")
    drive(store, sid, runtime=rt.child_runtime(store, sid))

    def fail(point):
        if point == cut:
            raise Restart

    with pytest.raises(Restart), store.atomic() as tx:
        child_delivery(store, tx, sid, hook=fail)
    assert not store.peek_inbox("control")
    store._fold_cache = None
    rt = runtime(store, database, configured(), config, llm_for(config))
    drive(store, sid, runtime=rt.child_runtime(store, sid))
    with store.atomic() as tx:
        assert child_delivery(store, tx, sid) > 0
    with store.atomic() as tx:
        assert child_delivery(store, tx, sid) == 0
    assert len(store.peek_inbox("control")) == 1
    drive(store, "control", runtime=rt)
    new = project_result(store, "control", "control/turn/initial", runtime=rt)
    assert tool_values(new) == tool_values(old) and new.final_output == old.final_output
