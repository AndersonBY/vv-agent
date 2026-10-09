"""F2d tools/control producers: public Runner paired with the durable kernel."""

import json
from copy import deepcopy
from dataclasses import replace
from threading import Event, Thread
from typing import Any

import pytest
from support import FixedModelProvider

from vv_agent import Agent, RunConfig, Runner
from vv_agent.approval import ApprovalBroker, ApprovalDecision
from vv_agent.llm.scripted import ScriptedLLM, ScriptStep
from vv_agent.run_config import ToolPolicy
from vv_agent.runtime.cancellation import CancelledError
from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeToolCallPatch
from vv_agent.session.kernel import Runtime, drive, read_state
from vv_agent.session.projection import project_records
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.tools.base import ToolContext
from vv_agent.tools.builtins import build_default_registry
from vv_agent.tools.executor import ToolExposure
from vv_agent.tools.function import function_tool
from vv_agent.types import AgentStatus, LLMResponse, ToolCall, ToolDirective, ToolExecutionResult

from .conftest import open_store
from .test_runner_parity import RESOLVED, echo, observe


class Restart(BaseException):
    pass


def admit(store, config, sid="control"):
    with store.atomic() as tx:
        tx.create(SessionSpec(sid, "test", str(config.workspace)), consumers=("events",))
        tx.push(sid, InboxItem("initial", "user", {"content": "go"}))


def runtime(store, database, agent, config, llm, **kwargs):
    return Runtime(agent, config, RESOLVED, llm, lambda: open_store(database), **kwargs)


@pytest.fixture
def database_clock(store, monkeypatch):
    now = 1_000_000
    monkeypatch.setattr(type(store), "clock_sql", str(now))

    def advance(milliseconds):
        nonlocal now
        now += milliseconds
        monkeypatch.setattr(type(store), "clock_sql", str(now))

    return advance


def records(store, sid="control", kind=None):
    return [r.record for r in read_state(store, sid)[1] if kind is None or r.record.kind == kind]


def reply(store, park, *, text="blue", input_id="reply", sid="control"):
    with store.atomic() as tx:
        tx.push(
            sid,
            InboxItem(
                input_id,
                "user",
                {
                    "content": {
                        "interaction_id": park.payload.get("interaction_id") or park.payload["handle"]["interaction_id"],
                        **({"operation_id": park.operation_id} if park.operation_id else {}),
                        "text": text,
                    }
                },
                park.turn_id,
                0,
            ),
        )


def approval_answer(store, park, decision="approve", input_id="answer"):
    h = park.payload["handle"]
    answer = InboxItem(
        input_id,
        "approval_answer",
        {
            "operation_id": park.operation_id,
            "attempt": park.attempt,
            "request_id": h["request_id"],
            "request_digest": h["request_digest"],
            "scope": h["scope"],
            "decision": decision,
        },
        park.turn_id,
        0,
    )
    with store.atomic() as tx:
        tx.push("control", answer)
    return answer


@function_tool
def finish(text: str) -> ToolExecutionResult:
    return ToolExecutionResult("", content=text, directive=ToolDirective.FINISH)


@pytest.mark.parametrize("visible", [False, True])
def test_hidden_tool_exposure_parity(tmp_path, visible):
    @function_tool(exposure=ToolExposure.DIRECT if visible else ToolExposure.HIDDEN)
    def secret() -> str:
        return "secret"

    def response(request):
        assert ("secret" in [s["function"]["name"] for s in request.tools]) == visible
        return LLMResponse("", [ToolCall("a", "secret", {})])

    config = RunConfig(workspace=tmp_path)
    old = observe(Agent("test", "test", tools=[secret]), [response, LLMResponse("done")], config, kernel=False)
    new = observe(Agent("test", "test", tools=[secret]), [response, LLMResponse("done")], config, kernel=True)
    assert new == old
    if not visible:
        assert "not allowed" in new["tools"][0][1]


def test_dynamic_schema_new_turn_parity(store, database, tmp_path):
    schema_seen = []

    def factory():
        registry = build_default_registry()
        registry.register_tool("custom", lambda ctx, args: ToolExecutionResult("", content="ok"), "first")
        return registry

    def first(request):
        schema_seen.append(next(s for s in request.tools if s["function"]["name"] == "custom")["function"]["description"])
        return LLMResponse("done")

    config = RunConfig(workspace=tmp_path, tool_registry_factory=factory)
    agent = Agent("test", "test")
    observe(agent, [first], config, kernel=False)
    config2 = replace(config, tool_registry_factory=lambda: changed_registry(factory()))
    observe(agent, [first], config2, kernel=False)
    admit(store, config)
    rt = runtime(store, database, agent, config, ScriptedLLM([first, first]))
    drive(store, "control", runtime=rt)
    changed_registry(rt.registry)
    with store.atomic() as tx:
        tx.push("control", InboxItem("second", "user", {"content": "go"}))
    drive(store, "control", runtime=rt)
    assert schema_seen == ["first", "second", "first", "second"]


def changed_registry(registry):
    schema = registry.get_schema("custom")
    schema["function"]["description"] = "second"
    registry.register_schema("custom", schema)
    return registry


def test_dynamic_schema_active_turn_rejected(store, database, tmp_path):
    config = RunConfig(workspace=tmp_path)
    admit(store, config)
    agent = Agent("test", "test")
    rt = runtime(
        store, database, agent, config, ScriptedLLM([LLMResponse("", [ToolCall("q", "ask_user", {"question": "Which?"})])])
    )
    drive(store, "control", runtime=rt)
    park = records(store, kind="op_parked")[0]
    schema = rt.registry.get_schema("ask_user")
    schema["function"]["description"] = "changed"
    rt.registry.register_schema("ask_user", schema)
    reply(store, park)
    drive(store, "control", runtime=rt)
    assert records(store, kind="turn_ended")[0].payload["reason"] == "handler_schema_or_capability_mismatch"


@pytest.mark.parametrize("no_tools", [False, True], ids=["ask_user", "no_tool"])
@pytest.mark.parametrize("pending", [False, True])
def test_user_wait_sdk_lifecycle_parity(store, database, tmp_path, no_tools, pending):
    calls = [] if no_tools else [ToolCall("q", "ask_user", {"question": "Which?", "options": ["blue", "red"]})]

    if pending and not no_tools:
        calls.append(ToolCall("pending", "finish", {"text": "must not execute"}))

    def answered(request):
        if pending and not no_tools:
            assert "Tool skipped because a previous tool requested user input." in next(
                m.content for m in request.messages if m.tool_call_id == "pending"
            )
        replies = [m.content for m in request.messages if m.role in {"user", "tool"}]
        assert "blue" in replies
        return LLMResponse("", [ToolCall("f", "finish", {"text": "blue"})])

    steps: list[ScriptStep] = [LLMResponse("Which?", calls), answered]
    agent = Agent("test", "test", tools=[finish])
    config = RunConfig(workspace=tmp_path, no_tool_policy="wait_user" if no_tools else "finish")
    runner = Runner.configured(replace(config, model_provider=FixedModelProvider(ScriptedLLM(deepcopy(steps)), RESOLVED)))
    waiting = runner.run_sync(agent, "go")
    assert waiting.status == AgentStatus.WAIT_USER
    resumed = runner.resume(waiting.into_state(), input="blue")
    assert resumed.final_output == "blue"
    admit(store, config)
    llm = ScriptedLLM(steps)
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    state = read_state(store, "control")[0]
    assert state.phase == "parked" and state.next_drive_ms is None
    park = records(store, kind="turn_parked" if no_tools else "op_parked")[0]
    if no_tools:
        assert not [r for r in records(store, kind="op_planned") if r.payload["op_kind"] != "model"]
    reply(store, park)
    reply(store, park, input_id="duplicate")
    reply(store, park, text="red", input_id="conflict")
    store._fold_cache = None
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    assert len(records(store, kind="turn_started")) == 1
    assert records(store, kind="turn_ended")[0].payload["result"] == resumed.final_output
    applied = records(store, kind="input_applied")
    assert [r.payload["disposition"] for r in applied] == ["queued", "applied", "noop", "rejected"]
    if not no_tools:
        events = project_records(read_state(store, "control")[1])
        assert [e.type for e in events].count("host_interaction_requested") == 1
        assert [e.type for e in events].count("host_interaction_response_consumed") == 1
        assert len([r for r in records(store, kind="op_completed") if r.operation_id == park.operation_id]) == 1
    # Late reply has no authority to start another turn.
    reply(store, park, input_id="late")
    drive(store, "control", runtime=runtime(store, database, agent, config, ScriptedLLM([])))
    assert records(store, kind="input_applied")[-1].payload["disposition"] == "noop"


@pytest.mark.parametrize(
    "policy",
    [
        ToolPolicy(),
        ToolPolicy(allowed_tools=["echo"]),
        ToolPolicy(allowed_tools=[]),
        ToolPolicy(disallowed_tools=["echo"]),
        ToolPolicy(can_use_tool=lambda name, args: args["text"] == "allowed"),
        ToolPolicy(denied_side_effects=["external"]),
        ToolPolicy(denied_capability_tags=["network"]),
        ToolPolicy(denied_cost_dimensions=["money"]),
        ToolPolicy(deny_terminal_tools=True),
    ],
    ids=["allow", "allow_list", "empty_list", "deny_list", "predicate", "side_effect", "tag", "cost", "terminal"],
)
@pytest.mark.parametrize("text", ["allowed", "denied"])
def test_tool_policy_matrix_parity(tmp_path, policy, text):
    from vv_agent.tools.metadata import ToolMetadata, ToolSideEffect

    tool = replace(
        echo,
        tool_metadata=ToolMetadata(
            side_effect=ToolSideEffect.EXTERNAL, capability_tags=["network"], cost_dimensions=["money"], terminal=True
        ),
    )
    agent = Agent("test", "test", tools=[tool])
    config = RunConfig(workspace=tmp_path, tool_policy=policy)
    steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", "echo", {"text": text})]), LLMResponse("done")]
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)


@pytest.mark.parametrize("frozen_denied", [False, True])
@pytest.mark.parametrize(
    "denial,source",
    [
        ("allowed_tools", "allowed_tools"),
        ("disallowed_tools", "disallowed_tools"),
        ("denied_side_effects", "metadata.side_effect"),
        ("denied_capability_tags", "metadata.capability_tag"),
        ("denied_cost_dimensions", "metadata.cost_dimension"),
        ("deny_terminal_tools", "metadata.terminal"),
    ],
)
def test_frozen_current_policy_dispatch_boundary(store, database, tmp_path, frozen_denied, denial, source):
    effects = []

    @function_tool(
        needs_approval=True,
        tool_metadata={"side_effect": "external", "capability_tags": ["network"], "cost_dimensions": ["money"], "terminal": True},
    )
    def effect() -> str:
        effects.append("effect")
        return "ok"

    def policy(denied):
        values: dict[str, Any] = {
            "allowed_tools": [] if denied else ["effect"],
            "disallowed_tools": ["effect"] if denied else [],
            "denied_side_effects": ["external"] if denied else [],
            "denied_capability_tags": ["network"] if denied else [],
            "denied_cost_dimensions": ["money"] if denied else [],
            "deny_terminal_tools": denied,
        }
        return ToolPolicy(**{denial: values[denial]})

    config = RunConfig(workspace=tmp_path, tool_policy=policy(frozen_denied))
    agent = Agent("test", "test", tools=[effect])
    admit(store, config)

    def cut(point, r):
        if point == "after_commit" and r.kind == "op_planned" and r.payload["op_kind"] != "model":
            raise Restart

    rt = runtime(store, database, agent, config, ScriptedLLM([LLMResponse("", [ToolCall("a", "effect", {})])]), hook=cut)
    with pytest.raises(Restart):
        drive(store, "control", runtime=rt)
    current = replace(config, tool_policy=policy(not frozen_denied))
    drive(store, "control", runtime=runtime(store, database, agent, current, ScriptedLLM([LLMResponse("done")])))
    assert effects == []
    result = next(r for r in records(store, kind="op_completed") if r.payload["result"].get("tool_call_id") == "a")
    assert result.payload["result"]["metadata"]["policy_source"] == source


class Decisions:
    def __init__(self, decision, request=True):
        self.decision, self.request, self.requests = decision, request, []

    def should_request(self, request):
        return self.request

    def decide(self, request):
        self.requests.append(request)
        return self.decision


@pytest.mark.parametrize("mode", ["default", "always", "never", "on_request"])
@pytest.mark.parametrize("needs", [False, True])
@pytest.mark.parametrize("action", ["allow", "deny", "allow_session", "timeout"])
def test_approval_mode_provider_parity(tmp_path, mode, needs, action):
    @function_tool(needs_approval=needs)
    def approved(text: str) -> str:
        return text

    def run(kernel):
        provider = Decisions(ApprovalDecision(action, reason="policy decision", metadata={"source": "test"}))
        config = RunConfig(workspace=tmp_path, tool_policy=ToolPolicy(approval=mode), approval_provider=provider)
        value = observe(
            Agent("test", "test", tools=[approved]),
            [
                LLMResponse("", [ToolCall("a", "approved", {"text": "one"}), ToolCall("b", "approved", {"text": "two"})]),
                LLMResponse("done"),
            ],
            config,
            kernel=kernel,
        )
        # Request IDs are independent transport identities inside denial metadata.
        value["tools"] = [(key, text) for key, text in value["tools"]]
        return value, len(provider.requests)

    assert run(True) == run(False)


def test_approval_broker_restart_session_and_conflicting_answer(store, database, tmp_path):
    @function_tool(needs_approval=True)
    def approved(text: str) -> str:
        return text

    agent = Agent("test", "test", tools=[approved])
    # SDK broker bridge using the same deferred provider protocol.
    broker = ApprovalBroker()

    class Auto(Decisions):
        def decide(self, request):
            self.requests.append(request)
            assert broker.resolve(request.request_id, "allow_session")
            return None

    old = observe(
        agent,
        [
            LLMResponse("", [ToolCall("a", "approved", {"text": "one"}), ToolCall("b", "approved", {"text": "two"})]),
            LLMResponse("done"),
        ],
        RunConfig(workspace=tmp_path, approval_provider=Auto(None), approval_broker=broker),
        kernel=False,
    )
    config = RunConfig(workspace=tmp_path, approval_provider=Decisions(None), approval_broker=ApprovalBroker())
    admit(store, config)
    llm = ScriptedLLM(
        [
            LLMResponse("", [ToolCall("a", "approved", {"text": "one"}), ToolCall("b", "approved", {"text": "two"})]),
            LLMResponse("done"),
        ]
    )
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    park = records(store, kind="op_parked")[0]
    assert not [r for r in records(store, kind="op_started") if r.operation_id == park.operation_id]
    assert config.approval_broker is not None
    assert config.approval_broker.resolve(park.payload["handle"]["request_id"], "allow_session")
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    result = records(store, kind="turn_ended")[0]
    assert result.payload["result"] == old["output"]
    assert len(records(store, kind="op_parked")) == 1
    # Replace both runtime and broker; session grants come from durable answers.
    with store.atomic() as tx:
        tx.push("control", InboxItem("next", "user", {"content": "go"}))
    drive(
        store,
        "control",
        runtime=runtime(
            store,
            database,
            agent,
            replace(config, approval_broker=ApprovalBroker()),
            ScriptedLLM([LLMResponse("", [ToolCall("c", "approved", {"text": "three"})]), LLMResponse("done")]),
        ),
    )
    assert len(records(store, kind="op_parked")) == 1
    retained = next(r for r in records(store, kind="input_applied") if r.payload["input"]["kind"] == "approval_answer")
    incoming = InboxItem(**retained.payload["input"])
    with store.atomic() as tx:
        tx.push("control", replace(incoming, input_id="duplicate"))
        tx.push("control", replace(incoming, input_id="conflict", payload=incoming.payload | {"decision": "deny"}))
    drive(store, "control", runtime=runtime(store, database, agent, config, ScriptedLLM([])))
    assert [r.payload["disposition"] for r in records(store, kind="input_applied")[-2:]] == ["noop", "rejected"]


def test_approval_timeout_restart_parity(store, database, tmp_path, database_clock):
    @function_tool(needs_approval=True)
    def approved() -> str:
        return "effect"

    agent = Agent("test", "test", tools=[approved])
    steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", "approved", {})]), LLMResponse("done")]
    old = observe(
        agent, steps, RunConfig(workspace=tmp_path, approval_provider=Decisions(None), approval_timeout_seconds=0), kernel=False
    )
    config = RunConfig(workspace=tmp_path, approval_provider=Decisions(None), approval_timeout_seconds=0.01)
    admit(store, config)
    llm = ScriptedLLM(steps)
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    # An expired durable deadline is evaluated by the new driver, not a fresh timeout.
    assert records(store, kind="op_parked")[0].payload["deadline_ms"] == 1_000_010
    database_clock(11)
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    results = [r.payload["result"] for r in records(store, kind="op_completed") if r.payload["result"].get("tool_call_id") == "a"]
    assert results[0]["error_code"] == "tool_approval_timeout"
    assert results[0]["content"] == old["tools"][0][1]
    assert records(store, kind="turn_ended")[0].payload["result"] == old["output"]


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("approval", [False, True])
@pytest.mark.parametrize("cut_kind", ["op_prepared", "op_completed"])
def test_tool_hooks_state_approval_restart_parity(store, database, tmp_path, short, approval, cut_kind):
    def setup():
        seen = []

        class Hook(BaseRuntimeHook):
            def before_tool_call(self, event):
                seen.append(("before", event.call.id))
                event.context.shared_state["count"] += 1
                patched = ToolCall(event.call.id, "effect", {"text": str(event.context.shared_state["count"])})
                return BeforeToolCallPatch(
                    call=patched, result=ToolExecutionResult(event.call.id, content="short") if short else None
                )

            def after_tool_call(self, event):
                seen.append(("after", event.call.id))
                event.context.shared_state["count"] += 10
                return replace(event.result, content=event.result.content + ":after")

        @function_tool(needs_approval=approval)
        def effect(text: str) -> str:
            seen.append(("effect", text))
            return text

        agent = Agent("test", "test", tools=[effect], hooks=[Hook()])
        config = RunConfig(workspace=tmp_path, shared_state={"count": 0}, approval_provider=Decisions(ApprovalDecision.allow()))
        return agent, config, seen

    steps: list[ScriptStep] = [
        LLMResponse("", [ToolCall("a", "effect", {"text": "raw"}), ToolCall("b", "effect", {"text": "raw"})]),
        LLMResponse("done"),
    ]
    old_agent, old_config, old_seen = setup()
    old = observe(old_agent, steps, old_config, kernel=False)
    agent, config, seen = setup()
    admit(store, config)
    llm = ScriptedLLM(steps)
    cut = False

    def stop(point, r):
        nonlocal cut
        if not cut and point == "after_commit" and r.kind == cut_kind and "/tool/" in (r.operation_id or ""):
            cut = True
            raise Restart

    with pytest.raises(Restart):
        drive(store, "control", runtime=runtime(store, database, agent, config, llm, hook=stop))
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    assert seen == old_seen
    tool_results = [
        (r.payload["result"]["tool_call_id"], r.payload["result"]["content"])
        for r in records(store, kind="op_completed")
        if "/tool/" in (r.operation_id or "")
    ]
    assert tool_results == old["tools"]
    assert records(store, kind="turn_ended")[0].payload["result"] == old["output"]
    assert records(store, kind="op_completed")[-1].payload["shared_state"] == old["shared"]


@pytest.mark.parametrize("behavior", ["run_llm_again", "stop_on_first_tool", "stop_at_tool_names"])
@pytest.mark.parametrize("native_finish", [False, True])
def test_tool_stop_pending_batch_native_finish_parity(tmp_path, behavior, native_finish):
    agent = Agent("test", "test", tools=[echo, finish], tool_use_behavior=behavior, stop_at_tool_names=["echo"])
    calls = [ToolCall("a", "finish" if native_finish else "echo", {"text": "done"}), ToolCall("b", "echo", {"text": "skipped"})]
    steps: list[ScriptStep] = [LLMResponse("", calls)] + (
        [LLMResponse("done")] if behavior == "run_llm_again" and not native_finish else []
    )
    old = observe(agent, steps, RunConfig(workspace=tmp_path), kernel=False)
    new = observe(agent, steps, RunConfig(workspace=tmp_path), kernel=True)
    # Runner's skipped pending results have no lifecycle events; kernel closes admitted plans durably.
    for value in (old, new):
        value.pop("tool_lifecycle")
        value["events"] = [v for v in value["events"] if v != "tool_call_completed"]
    assert new == old


@pytest.mark.parametrize("terminal", [False, True])
def test_background_process_restart_owner_parity(store, database, tmp_path, terminal):
    from vv_agent.runtime.background_sessions import background_session_manager as manager

    def scenario(kernel):
        sessions = []
        command = "sleep 0.05; printf finished" if terminal else "sleep 60"

        def manage(request):
            payload = json.loads(next(m.content for m in request.messages if m.tool_call_id == "bash"))
            sid = payload["session_id"]
            sessions.append(sid)
            if terminal:
                session = manager._get(sid)
                assert session is not None and session.done.wait(3)
            return LLMResponse(
                "",
                [
                    ToolCall("check", "check_background_command", {"session_id": sid}),
                    ToolCall("stop", "stop_background_command", {"session_id": sid}),
                ],
            )

        steps: list[ScriptStep] = [
            LLMResponse("", [ToolCall("bash", "bash", {"command": command, "yield_time_ms": 0})]),
            manage,
            LLMResponse("done"),
        ]
        config = RunConfig(workspace=tmp_path, tool_registry_factory=background_registry)
        agent = Agent("test", "test", tools=[])
        try:
            if not kernel:
                result = Runner.run_sync(
                    agent, "go", run_config=replace(config, model_provider=FixedModelProvider(ScriptedLLM(steps), RESOLVED))
                )
                values = [
                    normalized_process_result(r)
                    for c in result.raw_result.cycles
                    for r in c.tool_results
                    if r.tool_call_id != "bash"
                ]
                output = result.final_output
            else:
                admit(store, config)
                llm = ScriptedLLM(steps)

                def cut(point, record):
                    if (
                        point == "after_commit"
                        and record.kind == "op_completed"
                        and record.payload["result"].get("tool_call_id") == "bash"
                    ):
                        raise Restart

                with pytest.raises(Restart):
                    drive(store, "control", runtime=runtime(store, database, agent, config, llm, hook=cut))
                # Rebuild runtime and SQL fold; reuse only provider-owned process handles.
                store._fold_cache = None
                drive(store, "control", runtime=runtime(store, database, agent, config, llm))
                values = [
                    normalized_process_result(ToolExecutionResult.from_dict(r.payload["result"]))
                    for r in records(store, kind="op_completed")
                    if r.payload["result"].get("tool_call_id") in {"check", "stop"}
                ]
                output = records(store, kind="turn_ended")[0].payload["result"]
            for value in values:
                value.pop("session_id", None)
                value.pop("elapsed_seconds", None)
            assert values[0]["status"] == ("completed" if terminal else "running")
            assert values[1]["status"] == ("completed" if terminal else "stopped")
            return output, values
        finally:
            for sid in sessions:
                state = manager._get(sid)
                if state is not None:
                    assert state.artifact_backend is not None
                    manager.stop_for_tool(sid, state.artifact_backend, state.owner_task_id, "cleanup", workspace=tmp_path)
                    assert state.done.wait(3)

    assert scenario(True) == scenario(False)


@pytest.mark.parametrize("operation", ["check_background_command", "stop_background_command"])
def test_background_forbidden_owner_parity(tmp_path, operation):
    from vv_agent.runtime.background_sessions import background_session_manager as manager
    from vv_agent.workspace.local import LocalWorkspaceBackend

    backend = LocalWorkspaceBackend(tmp_path)
    sid = manager.start(command="sleep 60", cwd=tmp_path, owner_task_id="another-task", owner_workspace=tmp_path)
    try:
        steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", operation, {"session_id": sid})]), LLMResponse("done")]
        agent, config = Agent("test", "test", tools=[]), RunConfig(workspace=tmp_path, tool_registry_factory=background_registry)
        old, new = observe(agent, steps, config, kernel=False), observe(agent, steps, config, kernel=True)
        assert new == old
        assert "background_session_forbidden" in new["tools"][0][1]
        assert manager.check(sid)["status"] == "running"
    finally:
        assert manager.stop_for_tool(sid, backend, "another-task", "cleanup", workspace=tmp_path)["status"] == "stopped"


def test_background_unknown_stop_is_not_confirmed_parity(tmp_path, monkeypatch):
    import vv_agent.runtime.background_sessions as background

    def run(kernel):
        sessions = []

        def stop(request):
            sid = json.loads(next(m.content for m in request.messages if m.tool_call_id == "bash"))["session_id"]
            sessions.append(sid)
            monkeypatch.setattr(background, "kill_process_tree", lambda process: False)
            monkeypatch.setattr(background, "process_tree_is_running", lambda process: None)
            return LLMResponse("", [ToolCall("stop", "stop_background_command", {"session_id": sid})])

        try:
            value = observe(
                Agent("test", "test", tools=[]),
                [
                    LLMResponse("", [ToolCall("bash", "bash", {"command": "sleep 60", "yield_time_ms": 0})]),
                    stop,
                    LLMResponse("done"),
                ],
                RunConfig(workspace=tmp_path, tool_registry_factory=background_registry),
                kernel=kernel,
            )
            payload = json.loads(value["tools"][-1][1])
            assert payload["status"] in {"stopping", "unknown"}
            assert "exit_code" not in payload
            return payload["status"], value["output"]
        finally:
            monkeypatch.undo()
            for sid in sessions:
                session = background.background_session_manager._get(sid)
                assert session is not None and session.artifact_backend is not None
                background.background_session_manager.stop_for_tool(
                    sid, session.artifact_backend, session.owner_task_id, "cleanup", workspace=tmp_path
                )
                assert session.done.wait(3)

    assert run(True) == run(False)


@pytest.mark.parametrize("cooperative", [False, True], ids=["unknown", "cooperative"])
def test_cancellation_control_result_event_parity(store, database, tmp_path, cooperative):
    def setup():
        entered, release, exited = Event(), Event(), Event()

        @function_tool
        def blocked(ctx: ToolContext) -> str:
            entered.set()
            try:
                while not release.wait(0.005):
                    if cooperative:
                        assert ctx.ctx is not None
                        ctx.ctx.check_cancelled()
                return "late result"
            finally:
                exited.set()

        return Agent("test", "test", tools=[blocked]), entered, release, exited

    agent, entered, release, exited = setup()
    steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", "blocked", {})])]
    old_handle = Runner.start(
        agent,
        "go",
        run_config=RunConfig(workspace=tmp_path, model_provider=FixedModelProvider(ScriptedLLM(deepcopy(steps)), RESOLVED)),
    )
    try:
        assert entered.wait(3)
        assert old_handle.cancel()
        release.set()
        with pytest.raises(CancelledError):
            old_handle.result(timeout=3)
        old_events = list(old_handle.events())
        assert old_handle.state().cancelled
    finally:
        release.set()
        assert exited.wait(3)
    agent, entered, release, exited = setup()
    config = RunConfig(workspace=tmp_path, tool_registry_factory=background_registry)
    admit(store, config)
    errors = []

    def run():
        try:
            drive(
                store,
                "control",
                runtime=runtime(
                    store, database, agent, config, ScriptedLLM(deepcopy(steps)), heartbeat_seconds=0.01, cancellation_grace=0.01
                ),
            )
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=run)
    worker.start()
    try:
        assert entered.wait(3)
        with store.atomic() as tx:
            tx.push("control", InboxItem("cancel", "control", {"action": "cancel"}, "control/turn/initial"))
        worker.join(3)
        assert not worker.is_alive() and not errors
        end = records(store, kind="turn_ended")[0]
        assert end.payload["status"] == "cancelled"
        events = project_records(read_state(store, "control")[1])
        assert [e.type for e in events].count("run_cancelled") == 1
        assert [e.type for e in old_events].count("run_cancelled") == 0
        assert [e.type for e in events].count("tool_call_started") == [e.type for e in old_events].count("tool_call_started") == 1
        control = next(e for e in events if e.type == "run_state_changed")
        assert control.to_dict()["cancel_requested"] == {"from": False, "to": True}
        if cooperative:
            result = next(r for r in records(store, kind="op_completed") if r.payload["result"].get("tool_call_id") == "a")
            assert result.payload["result"]["error_code"] == "tool_cancelled"
            assert end.payload["unconfirmed_operations"] == []
        else:
            assert len(end.payload["unconfirmed_operations"]) == 1
            assert len(records(store, kind="op_unknown")) == 1
            assert any(e.type == "operation_ambiguous" for e in events)
    finally:
        release.set()
        worker.join(3)
        assert exited.wait(3)
    before = [r.digest for r in records(store)]
    drive(store, "control", runtime=runtime(store, database, agent, config, ScriptedLLM([])))
    assert [r.digest for r in records(store)] == before


def normalized_process_result(result):
    try:
        value = json.loads(result.content)
    except json.JSONDecodeError:
        value = {"output": result.content}
    return {"status": result.metadata["status"], "output": value.get("output", ""), "exit_code": result.metadata.get("exit_code")}


def background_registry():
    registry = build_default_registry()
    for name in ("bash", "check_background_command", "stop_background_command"):
        registry._add_planner_extra_tool_name(name)
    return registry


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("child_started", [False, True])
def test_cancellation_descendant_control_result_events_parity(store, database, tmp_path, background, child_started):
    from vv_agent.session.children import ChildSession

    entered, release = Event(), Event()

    def block(request):
        entered.set()
        assert release.wait(3)
        return LLMResponse("late child")

    child = Agent("child", "child")
    parent = Agent("parent", "parent", tools=[child.as_tool(name="delegate")])
    call = ToolCall("child", "delegate", {"task_description": "child"})
    handle = Runner.start(
        parent,
        "go",
        run_config=RunConfig(
            workspace=tmp_path, model_provider=FixedModelProvider(ScriptedLLM([LLMResponse("", [call]), block]), RESOLVED)
        ),
    )
    try:
        assert entered.wait(3)
        assert handle.cancel()
        release.set()
        # The current SDK child bridge also propagates the cancellation exception.
        with pytest.raises(CancelledError):
            handle.result(timeout=3)
        sdk_events = list(handle.events())
        assert handle.state().cancelled
        assert [e.type for e in sdk_events].count("tool_call_started") == 1
    finally:
        release.set()
        handle._thread.join(3)
        assert not handle._thread.is_alive()

    @function_tool
    def delegate(task_description: str) -> str:
        raise AssertionError("child admission must bypass the ordinary handler")

    config = RunConfig(workspace=tmp_path)
    agent = Agent("parent", "parent", tools=[delegate])
    admit(store, config)
    children = {
        "delegate": lambda plan: ChildSession(SessionSpec("child", "test", str(tmp_path)), "child", background=background)
    }
    steps: list[ScriptStep] = [LLMResponse("", [call])]
    if background:
        steps.append(LLMResponse("", [ToolCall("q", "ask_user", {"question": "Wait for cancellation?"})]))
    drive(store, "control", runtime=runtime(store, database, agent, config, ScriptedLLM(steps), children=children))
    child_entered, child_release, child_exited = Event(), Event(), Event()
    child_errors = []

    def waiting(request):
        child_entered.set()
        try:
            assert child_release.wait(3)
            return LLMResponse("late child")
        finally:
            child_exited.set()

    def run_child():
        try:
            with open_store(database) as child_store:
                drive(
                    child_store,
                    "child",
                    runtime=runtime(
                        child_store,
                        database,
                        Agent("child", "child"),
                        config,
                        ScriptedLLM([waiting]),
                        heartbeat_seconds=0.01,
                        cancellation_grace=0.01,
                    ),
                )
        except BaseException as exc:
            child_errors.append(exc)

    child_worker = Thread(target=run_child)
    if child_started:
        child_worker.start()
        assert child_entered.wait(3)
    try:
        with store.atomic() as tx:
            tx.push("control", InboxItem("cancel", "control", {"action": "cancel"}, "control/turn/initial"))
        drive(store, "control", runtime=runtime(store, database, agent, config, ScriptedLLM([]), children=children))
        assert records(store, kind="turn_ended")[0].payload["status"] == "cancelled"
        if child_started:
            child_worker.join(3)
            assert not child_worker.is_alive() and not child_errors
        else:
            drive(store, "child", runtime=runtime(store, database, Agent("child", "child"), config, ScriptedLLM([])))
        child_end = records(store, "child", "turn_ended")[0]
        assert child_end.payload["status"] == "cancelled"
        assert bool(records(store, "child", "op_started")) == child_started
        assert bool(child_end.payload["unconfirmed_operations"]) == child_started
    finally:
        child_release.set()
        if child_started:
            child_worker.join(3)
            assert child_exited.wait(3)
    for sid in ("control", "child"):
        events = project_records(read_state(store, sid)[1])
        assert any(e.type == "run_cancelled" for e in events)
    # Child cancellation never sends a cancel back to the parent.
    assert not store.peek_inbox("control")


@pytest.mark.parametrize("action", ["allow", "deny", "allow_session", "timeout"])
def test_provider_decision_receipt_events_restart(store, database, tmp_path, action):
    @function_tool(needs_approval=True)
    def effect() -> str:
        return "ok"

    provider = Decisions(ApprovalDecision(action, "reason", {"policy": "host"}))
    config = RunConfig(workspace=tmp_path, approval_provider=provider)
    agent = Agent("test", "test", tools=[effect])
    admit(store, config)
    llm = ScriptedLLM([LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")])

    def cut(point, r):
        if point == "after_commit" and r.kind == "input_applied" and r.payload["input"]["kind"] == "approval_answer":
            raise Restart

    with pytest.raises(Restart):
        drive(store, "control", runtime=runtime(store, database, agent, config, llm, hook=cut))
    assert len(provider.requests) == 1
    replacement = Decisions(ApprovalDecision.deny("must not be invoked"))
    drive(store, "control", runtime=runtime(store, database, agent, replace(config, approval_provider=replacement), llm))
    assert replacement.requests == []
    answer = next(r for r in records(store, kind="input_applied") if r.payload["input"]["kind"] == "approval_answer")
    assert answer.payload["input"]["payload"]["metadata"] == {"policy": "host"}
    events = project_records(read_state(store, "control")[1])
    approvals = [e for e in events if e.type in {"approval_requested", "approval_resolved"}]
    assert [e.type for e in approvals] == ["approval_requested", "approval_resolved"]
    assert approvals[-1].to_dict()["action"] == action
    assert approvals[-1].metadata["reason"] == "reason"
    assert approvals[-1].metadata["decision_metadata"] == {"policy": "host"}


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("native_finish", [False, True])
@pytest.mark.parametrize("behavior", ["run_llm_again", "stop_on_first_tool", "stop_at_tool_names"])
def test_tool_stop_error_result_parity(tmp_path, success, native_finish, behavior):
    from vv_agent.types import ToolResultStatus

    @function_tool
    def maybe() -> ToolExecutionResult:
        return ToolExecutionResult(
            "",
            content="first",
            status_code=ToolResultStatus.SUCCESS if success else ToolResultStatus.ERROR,
            directive=ToolDirective.FINISH if native_finish else ToolDirective.CONTINUE,
        )

    agent = Agent("test", "test", tools=[maybe, echo], tool_use_behavior=behavior, stop_at_tool_names=["maybe", "echo"])
    steps: list[ScriptStep] = [
        LLMResponse(
            "", [ToolCall("a", "maybe", {}), ToolCall("b", "echo", {"text": "second"}), ToolCall("c", "echo", {"text": "third"})]
        )
    ]
    if behavior == "run_llm_again" and not native_finish:
        steps.append(LLMResponse("done"))
    config = RunConfig(workspace=tmp_path)
    old, new = observe(agent, steps, config, kernel=False), observe(agent, steps, config, kernel=True)
    assert new["output"] == old["output"] and new["tools"] == old["tools"] and new["shared"] == old["shared"]
    assert new["events"].count("tool_call_started") == old["events"].count("tool_call_started")


@pytest.mark.parametrize("key,value", [("reason", 1), ("metadata", []), ("extra", True)])
def test_approval_answer_closed_optional_fields(key, value):
    from vv_agent.session.records import RecordError, digest

    payload = {
        "operation_id": "op",
        "attempt": 1,
        "request_id": "req",
        "request_digest": digest({}),
        "scope": ["echo"],
        "decision": "approve",
        key: value,
    }
    with pytest.raises(RecordError):
        InboxItem("answer", "approval_answer", payload).encode()


@pytest.mark.parametrize("should_request", [False, True])
def test_approval_provider_metadata_and_request_boundary_parity(tmp_path, should_request):
    def run(kernel):
        class Provider(Decisions):
            def should_request(self, request):
                assert request.metadata["tool_metadata"]["rule"] == "host"
                return self.request

        @function_tool(needs_approval=True)
        def effect() -> str:
            return "ok"

        effect.metadata["rule"] = "host"
        provider = Provider(ApprovalDecision.allow(), request=should_request)
        value = observe(
            Agent("test", "test", tools=[effect]),
            [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
            RunConfig(workspace=tmp_path, approval_provider=provider),
            kernel=kernel,
        )
        return value, len(provider.requests)

    assert run(True) == run(False)


@pytest.mark.parametrize("decision", ["approve", "allow_session"])
def test_expired_approval_answer_cannot_authorize_effect(store, database, tmp_path, decision, database_clock):
    effects = []

    @function_tool(needs_approval=True)
    def effect() -> str:
        effects.append("effect")
        return "effect"

    config = RunConfig(workspace=tmp_path, approval_provider=Decisions(None), approval_timeout_seconds=0.02)
    agent = Agent("test", "test", tools=[effect])
    admit(store, config)
    llm = ScriptedLLM([LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")])
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    park = records(store, kind="op_parked")[0]
    database_clock(21)
    approval_answer(store, park, decision=decision)
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    assert effects == []
    late = next(r for r in records(store, kind="input_applied") if r.payload["input"]["input_id"] == "answer")
    assert late.payload["disposition"] == "rejected"
    result = next(r for r in records(store, kind="op_completed") if r.payload["result"].get("tool_call_id") == "a")
    assert result.payload["result"]["error_code"] == "tool_approval_timeout"
    # SDK's already timed-out request is no longer resolvable either.
    broker = ApprovalBroker()

    class Capture(Decisions):
        def decide(self, request):
            self.requests.append(request)
            return None

    capture = Capture(None)
    observe(
        agent,
        [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")],
        replace(config, approval_provider=capture, approval_broker=broker, approval_timeout_seconds=0),
        kernel=False,
    )
    assert not broker.resolve(capture.requests[0].request_id, "allow")
    assert effects == []


def test_rejected_broker_session_grant_cannot_authorize_next_tool(store, database, tmp_path, database_clock):
    effects = []

    @function_tool(needs_approval=True)
    def effect() -> str:
        effects.append("effect")
        return "effect"

    broker = ApprovalBroker()
    config = RunConfig(
        workspace=tmp_path, approval_provider=Decisions(None), approval_broker=broker, approval_timeout_seconds=0.02
    )
    agent = Agent("test", "test", tools=[effect])
    admit(store, config)
    llm = ScriptedLLM([LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")])
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    park = records(store, kind="op_parked")[0]
    database_clock(21)
    assert broker.resolve(park.payload["handle"]["request_id"], "allow_session")
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    assert effects == []
    assert broker.is_session_allowed("effect")
    with store.atomic() as tx:
        tx.push("control", InboxItem("next", "user", {"content": "go"}))
    drive(
        store,
        "control",
        runtime=runtime(
            store,
            database,
            agent,
            replace(config, approval_timeout_seconds=10),
            ScriptedLLM([LLMResponse("", [ToolCall("b", "effect", {})])]),
        ),
    )
    assert effects == []
    assert len(records(store, kind="op_parked")) == 2
    assert not [r for r in records(store, kind="op_started") if "/tool/" in (r.operation_id or "")]


@pytest.mark.parametrize("initially_allowed", [False, True])
def test_current_policy_predicate_rechecked_after_restart(store, database, tmp_path, initially_allowed):
    effects = []

    @function_tool
    def effect() -> str:
        effects.append("effect")
        return "effect"

    allowed = initially_allowed

    def response(request):
        nonlocal allowed
        allowed = not initially_allowed
        return LLMResponse("", [ToolCall("a", "effect", {})])

    agent = Agent("test", "test", tools=[effect])
    config = RunConfig(workspace=tmp_path, tool_policy=ToolPolicy(can_use_tool=lambda name, args: allowed))
    old = observe(agent, [response, LLMResponse("done")], config, kernel=False)
    old_effects = list(effects)
    effects.clear()
    allowed = initially_allowed
    admit(store, config)

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_planned" and record.payload["op_kind"] != "model":
            raise Restart

    llm = ScriptedLLM([LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")])
    with pytest.raises(Restart):
        drive(store, "control", runtime=runtime(store, database, agent, config, llm, hook=cut))
    current = replace(config, tool_policy=ToolPolicy(can_use_tool=lambda name, args: not initially_allowed))
    store._fold_cache = None
    drive(store, "control", runtime=runtime(store, database, agent, current, llm))
    result = next(
        r.payload["result"] for r in records(store, kind="op_completed") if r.payload["result"].get("tool_call_id") == "a"
    )
    assert [(result["tool_call_id"], result["content"])] == old["tools"]
    assert effects == old_effects == ([] if initially_allowed else ["effect"])
    assert records(store, kind="turn_ended")[0].payload["result"] == old["output"]
    if initially_allowed:
        assert result["metadata"]["policy_source"] == "can_use_tool"


def test_approval_absolute_deadline_includes_provider_time(store, database, tmp_path):
    effects = []

    @function_tool(needs_approval=True)
    def effect() -> str:
        effects.append("effect")
        return "effect"

    agent = Agent("test", "test", tools=[effect])
    provider = Decisions(ApprovalDecision.allow())
    config = RunConfig(workspace=tmp_path, approval_provider=provider, approval_timeout_seconds=0)
    steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", "effect", {})]), LLMResponse("done")]
    old = observe(agent, steps, config, kernel=False)
    assert old["tools"] == [("a", "effect")] and effects == ["effect"]
    assert len(provider.requests) == 1
    effects.clear()
    provider = Decisions(ApprovalDecision.allow())
    config = replace(config, approval_provider=provider)
    admit(store, config)
    drive(store, "control", runtime=runtime(store, database, agent, config, ScriptedLLM(steps)))
    result = next(
        r.payload["result"] for r in records(store, kind="op_completed") if r.payload["result"].get("tool_call_id") == "a"
    )
    assert effects == [] and provider.requests == []
    assert result["error_code"] == "tool_approval_timeout"
    assert records(store, kind="turn_ended")[0].payload["result"] == old["output"]
    approvals = [e for e in project_records(read_state(store, "control")[1]) if e.type.startswith("approval_")]
    assert [e.type for e in approvals] == ["approval_requested", "approval_resolved"]
    assert approvals[-1].to_dict()["action"] == "timeout"


@pytest.mark.parametrize("visible", [False, True])
def test_hook_patch_preserves_hidden_tool_boundary(store, database, tmp_path, visible):
    def setup():
        seen = []

        @function_tool(exposure=ToolExposure.DIRECT if visible else ToolExposure.HIDDEN)
        def secret() -> str:
            seen.append("effect")
            return "secret"

        class Hook(BaseRuntimeHook):
            def before_tool_call(self, event):
                seen.append("before")
                return BeforeToolCallPatch(call=ToolCall(event.call.id, "secret", {}))

            def after_tool_call(self, event):
                seen.append("after")
                return event.result

        return Agent("test", "test", tools=[echo, secret], hooks=[Hook()]), seen

    steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", "echo", {"text": "original"})]), LLMResponse("done")]
    old_agent, old_seen = setup()
    config = RunConfig(workspace=tmp_path)
    old = observe(old_agent, steps, config, kernel=False)
    agent, seen = setup()
    admit(store, config)
    llm = ScriptedLLM(steps)

    def cut(point, record):
        if point == "after_commit" and record.kind == "op_prepared":
            raise Restart

    with pytest.raises(Restart):
        drive(store, "control", runtime=runtime(store, database, agent, config, llm, hook=cut))
    store._fold_cache = None
    drive(store, "control", runtime=runtime(store, database, agent, config, llm))
    result = next(
        r.payload["result"] for r in records(store, kind="op_completed") if r.payload["result"].get("tool_call_id") == "a"
    )
    assert seen == old_seen == (["before", "effect", "after"] if visible else ["before", "after"])
    assert [(result["tool_call_id"], result["content"])] == old["tools"]
    assert records(store, kind="turn_ended")[0].payload["result"] == old["output"]
    if not visible:
        assert result["error_code"] == "tool_not_allowed"
        assert not [r for r in records(store, kind="op_started") if "/tool/" in (r.operation_id or "")]
