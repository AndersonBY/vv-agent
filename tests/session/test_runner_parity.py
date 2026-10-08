"""Same scripted scenarios on the public Runner and the internal SQLite kernel."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace

import pytest
from support import FixedModelProvider

from vv_agent import Agent, ModelSettings, RunConfig, Runner
from vv_agent.budget import RunBudgetLimits
from vv_agent.config import ResolvedModelConfig
from vv_agent.llm.scripted import ScriptedLLM, ScriptStep
from vv_agent.session.context import project_context
from vv_agent.session.kernel import Runtime, drive, read_state
from vv_agent.session.projection import project_records
from vv_agent.session.records import InboxItem, SessionSpec
from vv_agent.session.sqlite import SQLiteStore
from vv_agent.tools.function import function_tool
from vv_agent.types import LLMResponse, ToolCall

RESOLVED = ResolvedModelConfig("scripted", "m", "m", "m", [])
COMMON_EVENTS = {
    "run_started",
    "model_call_started",
    "model_call_completed",
    "tool_call_started",
    "tool_call_completed",
    "run_completed",
    "run_failed",
    "run_cancelled",
}


def observe(agent, steps, config, *, kernel, provider_settings=None):
    provider = FixedModelProvider(ScriptedLLM(deepcopy(steps)), RESOLVED, settings=provider_settings or ModelSettings())
    config = replace(config, model_provider=provider)
    if not kernel:
        result = Runner.run_sync(agent, "go", run_config=config)
        return {
            "output": result.final_output,
            "tools": [(m.tool_call_id, m.content) for m in result.raw_result.messages if m.role == "tool"],
            "events": [e.type for e in result.events if e.type in COMMON_EVENTS],
            "tool_lifecycle": tool_lifecycle(result.events),
            "budget": normalized_budget(result.budget_usage.to_dict() if result.budget_usage else {}),
            "shared": result.raw_result.shared_state,
        }
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("parity", "test", str(config.workspace)), consumers=("events",))
            tx.push("parity", InboxItem("input", "user", {"content": "go"}))
        rt = Runtime(agent, config, RESOLVED, provider.llm, lambda: nullcontext(store))
        drive(store, "parity", runtime=rt)
        state, records, _ = read_state(store, "parity")
        terminal = next(r.record.payload for r in reversed(records) if r.record.kind == "turn_ended")
        snapshots = [r.record.payload["usage"].get("session_shared_state") for r in records if r.record.kind == "op_completed"]
        return {
            "output": terminal["result"],
            "tools": [(m.tool_call_id, m.content) for m in project_context(records, state) if m.role == "tool"],
            "events": [e.type for e in project_records(records) if e.type in COMMON_EVENTS],
            "tool_lifecycle": tool_lifecycle(project_records(records)),
            "budget": normalized_budget(terminal["budget"]),
            "shared": next((v for v in reversed(snapshots) if v is not None), config.shared_state or {}),
        }


def tool_lifecycle(events):
    result = {}
    for event in events:
        if event.type in {"tool_call_planned", "tool_call_started", "tool_call_completed"}:
            result.setdefault(event.tool_call_id, []).append(event.type)
    return result


def normalized_budget(value):
    return {k: v for k, v in value.items() if k != "elapsed_ms"}


@function_tool
def echo(text: str) -> str:
    return text


@pytest.mark.parametrize("calls", [[], [ToolCall("a", "echo", {"text": "one"}), ToolCall("b", "echo", {"text": "two"})]])
def test_basic_runner_parity(tmp_path, calls):
    agent = Agent("parity", "Be concise.", tools=[echo])
    steps = ([LLMResponse("", calls)] if calls else []) + [LLMResponse("done")]
    config = RunConfig(workspace=tmp_path)
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)


@pytest.mark.parametrize("enabled", [True, False])
def test_dynamic_enabled_and_registry_factory_parity(tmp_path, enabled):
    from vv_agent.tools.builtins import build_default_registry

    @function_tool(is_enabled=lambda ctx, agent: ctx.app_state["enabled"])
    def conditional() -> str:
        return "enabled"

    def factory():
        registry = build_default_registry()
        registry.register_executor(echo.to_executor(), planner_extra=True)
        return registry

    def response(request):
        names = [s["function"]["name"] for s in request.tools]
        assert ("conditional" in names) == enabled
        assert "echo" in names
        return LLMResponse("done")

    config = RunConfig(workspace=tmp_path, context={"enabled": enabled}, tool_registry_factory=factory)
    agent = Agent("parity", "Be concise.", tools=[conditional])
    assert observe(agent, [response], config, kernel=True) == observe(agent, [response], config, kernel=False)


def test_shared_state_and_settings_parity(tmp_path):
    from vv_agent.tools.base import ToolContext

    @function_tool
    def increment(ctx: ToolContext) -> str:
        ctx.shared_state["count"] += 1
        return str(ctx.shared_state["count"])

    def response(request):
        assert request.model_settings.temperature == 0.2
        assert request.model_settings.max_tokens == 120
        return LLMResponse("done")

    agent = Agent("parity", "Be concise.", tools=[increment], model_settings=ModelSettings(temperature=0.2, max_tokens=50))
    config = RunConfig(workspace=tmp_path, shared_state={"count": 0}, model_settings=ModelSettings(max_tokens=120))
    steps = [LLMResponse("", [ToolCall("a", "increment", {}), ToolCall("b", "increment", {})]), response]
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)


def test_budget_usage_parity(tmp_path):
    agent = Agent("parity", "Be concise.", tools=[echo])
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "prompt_tokens_details": {"cached_tokens": 0}}
    steps = [
        LLMResponse("", [ToolCall("a", "echo", {"text": "one"})], raw={"usage": usage}),
        LLMResponse("done", raw={"usage": usage}),
    ]
    config = RunConfig(workspace=tmp_path, budget_limits=RunBudgetLimits(max_total_tokens=1000, max_tool_calls=5))
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)


@pytest.mark.parametrize("stage", ["input", "output"])
@pytest.mark.parametrize("action", ["allow", "rewrite", "block", "require_approval"])
def test_guardrail_parity(tmp_path, stage, action):
    from vv_agent.guardrails import GuardrailResult

    def guard(ctx, value):
        assert ctx.app_state == {"tenant": "test"}
        return GuardrailResult(action, message="blocked", value="rewritten")

    agent = Agent("parity", "Be concise.")
    setattr(agent, f"{stage}_guardrails", [guard])
    config = RunConfig(workspace=tmp_path, context={"tenant": "test"})
    old = observe(agent, [LLMResponse("done")], config, kernel=False)
    new = observe(agent, [LLMResponse("done")], config, kernel=True)
    # Runner blocks before compiling initial state; kernel retains the frozen definition.
    if stage == "input" and action in {"block", "require_approval"}:
        old.pop("shared")
        new.pop("shared")
    assert new == old


@pytest.mark.parametrize("repair", [False, True])
def test_output_validation_parity(tmp_path, repair):
    from vv_agent.output_validation import OutputValidationResult

    repairs = []

    def fix(request):
        assert request.tools == ()
        repairs.append(request.invalid_output)
        return "valid"

    agent = Agent(
        "parity",
        "Be concise.",
        output_validation_enabled=True,
        output_validator=lambda value, ctx: (
            OutputValidationResult.accept() if value == "valid" else OutputValidationResult.reject("invalid", "bad answer")
        ),
        output_repair=fix if repair else None,
    )
    config = RunConfig(workspace=tmp_path)
    old = observe(agent, [LLMResponse("bad")], config, kernel=False)
    new = observe(agent, [LLMResponse("bad")], config, kernel=True)
    if repair:
        # The old callback has no model ledger entry; kernel makes repair dispatch durable.
        assert new["events"].count("model_call_started") == 2
        new["events"] = new["events"][:3] + new["events"][5:]
        assert repairs == ["bad", "bad"]
    assert new == old


def test_llm_hook_parity(tmp_path):
    from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeLLMPatch
    from vv_agent.types import Message

    class Hook(BaseRuntimeHook):
        def before_llm(self, event):
            event.shared_state["hook"] = "before"
            return BeforeLLMPatch(messages=[*event.messages, Message("user", "hook input")])

        def after_llm(self, event):
            assert event.shared_state["hook"] == "before"
            return LLMResponse("hook output")

    def response(request):
        assert request.messages[-1].content == "hook input"
        return LLMResponse(
            "raw",
            raw={
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 0},
                }
            },
        )

    agent = Agent("parity", "Be concise.", hooks=[Hook()])
    config = RunConfig(workspace=tmp_path, budget_limits=RunBudgetLimits(max_total_tokens=100))
    assert observe(agent, [response], config, kernel=True) == observe(agent, [response], config, kernel=False)


@pytest.mark.parametrize("behavior", ["stop_on_first_tool", "stop_at_tool_names"])
def test_tool_stop_parity(tmp_path, behavior):
    agent = Agent("parity", "Be concise.", tools=[echo], tool_use_behavior=behavior, stop_at_tool_names=["echo"])
    config = RunConfig(workspace=tmp_path)
    steps = [LLMResponse("", [ToolCall("a", "echo", {"text": "stopped"})])]
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)


def test_tool_hooks_parity(tmp_path):
    from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeToolCallPatch
    from vv_agent.types import ToolExecutionResult

    class Hook(BaseRuntimeHook):
        def before_tool_call(self, event):
            return BeforeToolCallPatch(call=ToolCall(event.call.id, "echo", {"text": "patched"}))

        def after_tool_call(self, event):
            return ToolExecutionResult(event.call.id, content=event.result.content + " after")

    agent = Agent("parity", "Be concise.", tools=[echo], hooks=[Hook()])
    config = RunConfig(workspace=tmp_path)
    steps = [LLMResponse("", [ToolCall("a", "echo", {"text": "original"})]), LLMResponse("done")]
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)


def test_context_provider_parity(tmp_path):
    from vv_agent.context_providers import ContextFragment

    class Provider:
        def fragments(self, request):
            return [ContextFragment(id="host", text="host context", stable=False)]

    def response(request):
        assert "host context" in request.prompt_bundle.to_dict()["sections"][1]["text"]
        return LLMResponse("done")

    config = RunConfig(workspace=tmp_path, context_providers=[Provider()])
    agent = Agent("parity", "Be concise.")
    assert observe(agent, [response], config, kernel=True) == observe(agent, [response], config, kernel=False)


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("read_file", {"path": "sample.txt"}),
        ("write_file", {"path": "new.txt", "content": "created"}),
        ("edit_file", {"path": "sample.txt", "old_string": "original", "new_string": "changed"}),
        ("find_files", {"glob": "*.txt"}),
        ("search_files", {"pattern": "original"}),
    ],
)
def test_workspace_tool_parity(tmp_path, name, arguments):
    import json

    from vv_agent.run_config import ToolPolicy

    def one(kernel):
        (tmp_path / "sample.txt").write_text("original\n")
        (tmp_path / "new.txt").unlink(missing_ok=True)
        steps = []
        if name == "edit_file":
            steps.append(LLMResponse("", [ToolCall("read", "read_file", {"path": "sample.txt"})]))
        steps.extend([LLMResponse("", [ToolCall("a", name, arguments)]), LLMResponse("done")])
        result = observe(
            Agent("parity", "Be concise."),
            steps,
            RunConfig(workspace=tmp_path, tool_policy=ToolPolicy(approval="never")),
            kernel=kernel,
        )
        # Tool receipts contain file modification times; data and error/success fields must match.
        for _, content in result["tools"]:
            if content.startswith("{"):
                value = json.loads(content)
                assert not value.get("error"), content
        return result

    assert one(True) == one(False)


def test_endpoint_attempts_do_not_stack(tmp_path, monkeypatch):
    from vv_agent.llm.vv_llm_client import EndpointTarget, VvLlmClient

    calls = []

    def complete(client, request):
        assert len(client.endpoint_targets) == 1
        assert client.max_retries_per_endpoint == 1
        assert request.model_settings.retry.max_attempts == 1
        calls.append(client.endpoint_targets[0].endpoint_id)
        if len(calls) == 1:
            raise TimeoutError("receipt lost")
        return LLMResponse("done")

    monkeypatch.setattr(VvLlmClient, "complete", complete)
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "test", str(tmp_path)), consumers=())
            tx.push("s", InboxItem("i", "user", {"content": "go"}))
        llm = VvLlmClient(
            [EndpointTarget("a", "unused", "https://example.invalid"), EndpointTarget("b", "unused", "https://example.invalid")],
            randomize_endpoints=False,
        )
        runtime = Runtime(
            Agent("parity", "Be concise."), RunConfig(workspace=tmp_path), RESOLVED, llm, lambda: nullcontext(store)
        )
        drive(store, "s", runtime=runtime)
        state, records, _ = read_state(store, "s")
        assert state.active_turn_id is None
        assert records[-1].record.payload["result"] == "done"
        assert calls == ["a", "b"]
        assert llm.max_retries_per_endpoint == 3 and len(llm.endpoint_targets) == 2


def test_shared_state_survives_runtime_reconstruction(tmp_path):
    from vv_agent.tools.base import ToolContext

    @function_tool
    def increment(ctx: ToolContext) -> str:
        ctx.shared_state["count"] += 1
        return str(ctx.shared_state["count"])

    class StopDrive(BaseException):
        pass

    def stop(point, record):
        if point == "after_commit" and record.kind == "op_completed" and "/tool/" in record.operation_id:
            raise StopDrive

    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "test", str(tmp_path)), consumers=())
            tx.push("s", InboxItem("i", "user", {"content": "go"}))
        agent = Agent("parity", "Be concise.", tools=[increment])
        config = RunConfig(workspace=tmp_path, shared_state={"count": 0})
        first = Runtime(
            agent,
            config,
            RESOLVED,
            ScriptedLLM(
                [
                    LLMResponse("", [ToolCall("a", "increment", {}), ToolCall("b", "increment", {})]),
                ]
            ),
            lambda: nullcontext(store),
            hook=stop,
        )
        with pytest.raises(StopDrive):
            drive(store, "s", runtime=first)
        store._fold_cache = None
        second = Runtime(agent, config, RESOLVED, ScriptedLLM([LLMResponse("done")]), lambda: nullcontext(store))
        drive(store, "s", runtime=second)
        state, records, _ = read_state(store, "s")
        assert [m.content for m in project_context(records, state) if m.role == "tool"] == ["1", "2"]


def test_blocked_input_does_not_compile_providers(tmp_path):
    from vv_agent.guardrails import GuardrailResult

    def instructions(ctx, agent):
        raise AssertionError("blocked input reached instruction provider")

    agent = Agent("parity", instructions, input_guardrails=[lambda ctx, value: GuardrailResult.block("blocked")])
    config = RunConfig(workspace=tmp_path)
    assert observe(agent, [], config, kernel=True) == observe(agent, [], config, kernel=False)


def test_result_projection_parity(tmp_path):
    from vv_agent.session.result import project_result

    agent = Agent("parity", "Be concise.", tools=[echo])
    steps: list[ScriptStep] = [LLMResponse("", [ToolCall("a", "echo", {"text": "one"})]), LLMResponse("done")]
    config = RunConfig(workspace=tmp_path)
    old = Runner.run_sync(
        agent, "go", run_config=replace(config, model_provider=FixedModelProvider(ScriptedLLM(deepcopy(steps)), RESOLVED))
    )
    with SQLiteStore.standalone(":memory:") as store:
        store.install_schema()
        with store.atomic() as tx:
            tx.create(SessionSpec("s", "test", str(tmp_path)), consumers=())
            tx.push("s", InboxItem("i", "user", {"content": "go"}))
        runtime = Runtime(agent, config, RESOLVED, ScriptedLLM(deepcopy(steps)), lambda: nullcontext(store))
        drive(store, "s", runtime=runtime)
        new = project_result(store, "s", "s/turn/i", runtime=runtime)
        for name in (
            "input",
            "final_output",
            "status",
            "agent_name",
            "completion_reason",
            "completion_tool_name",
            "partial_output",
            "wait_reason",
            "budget_usage",
            "budget_exhaustion",
        ):
            assert getattr(new, name) == getattr(old, name), name
        assert new.raw_result.shared_state == old.raw_result.shared_state
        assert [c.to_dict() for c in new.raw_cycles] == [c.to_dict() for c in old.raw_cycles]
        assert [(m.role, m.content) for m in new.new_items] == [(m.role, m.content) for m in old.new_items]
        assert new.resolved_model == old.resolved_model
        for call in new.token_usage.model_calls:
            assert "session_shared_state" not in call.usage.provider_usage


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("file_info", {"path": "sample.txt"}),
        ("bash", {"command": "printf kernel-parity"}),
        ("read_image", {"path": "https://example.invalid/test.png"}),
    ],
)
def test_additional_builtin_parity(tmp_path, name, arguments):
    from vv_agent.tools.builtins import build_default_registry
    from vv_agent.tools.registry import ToolRegistry

    (tmp_path / "sample.txt").write_text("original\n")
    tool = build_default_registry().get_executor(name)
    agent = Agent("parity", "Be concise.", tools=[tool])
    steps = [LLMResponse("", [ToolCall("a", name, arguments)]), LLMResponse("done")]
    config = RunConfig(workspace=tmp_path, tool_registry_factory=ToolRegistry)
    new, old = observe(agent, steps, config, kernel=True), observe(agent, steps, config, kernel=False)
    assert new == old
    if name == "bash":
        assert new["tools"] == [("a", "kernel-parity")]


def test_todo_and_skill_state_parity(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    import vv_agent.tools.handlers.todo as todo_module

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 8, tzinfo=UTC)

    monkeypatch.setattr(todo_module, "datetime", Clock)
    agent = Agent(
        "parity",
        "Be concise.",
        metadata={
            "available_skills": [
                {"name": "inspect", "description": "Inspect carefully", "instructions": "Read first."},
            ]
        },
    )
    steps = [
        LLMResponse(
            "",
            [
                ToolCall(
                    "a",
                    "todo_write",
                    {
                        "todos": [
                            {"id": "todo-1", "title": "Inspect", "status": "in_progress", "priority": "medium"},
                        ]
                    },
                ),
                ToolCall("b", "activate_skill", {"skill_name": "inspect", "reason": "inspect"}),
            ],
        ),
        LLMResponse("done"),
    ]
    config = RunConfig(workspace=tmp_path)
    new, old = observe(agent, steps, config, kernel=True), observe(agent, steps, config, kernel=False)
    assert new == old
    assert new["shared"]["active_skills"] == ["inspect"]
    assert new["shared"]["todo_list"][0]["title"] == "Inspect"


def test_memory_workspace_backend_parity(tmp_path):
    from vv_agent.workspace.memory import MemoryWorkspaceBackend

    backend = MemoryWorkspaceBackend()
    backend.write_text("sample.txt", "memory backend")
    steps = [LLMResponse("", [ToolCall("a", "read_file", {"path": "sample.txt"})]), LLMResponse("done")]
    agent = Agent("parity", "Be concise.")
    config = RunConfig(workspace=tmp_path, workspace_backend=backend)
    new, old = observe(agent, steps, config, kernel=True), observe(agent, steps, config, kernel=False)
    assert new == old
    assert "memory backend" in new["tools"][0][1]
    assert not (tmp_path / "sample.txt").exists()


def test_llm_hook_tool_visibility_is_enforced(tmp_path):
    from vv_agent.runtime.hooks import BaseRuntimeHook, BeforeLLMPatch

    class Hook(BaseRuntimeHook):
        def before_llm(self, event):
            return BeforeLLMPatch(tool_schemas=[])

    effects = []

    @function_tool
    def hidden() -> str:
        effects.append("executed")
        return "unsafe"

    agent = Agent("parity", "Be concise.", tools=[hidden], hooks=[Hook()])
    steps = [LLMResponse("", [ToolCall("a", "hidden", {})]), LLMResponse("done")]
    config = RunConfig(workspace=tmp_path)
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)
    assert effects == []


@pytest.mark.parametrize("image_path", ["https://example.invalid/tool.png", "pixel.png"])
def test_multimodal_model_context_parity(tmp_path, monkeypatch, image_path):
    import base64

    (tmp_path / "pixel.png").write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j1XkAAAAASUVORK5CYII=")
    )
    from vv_agent.types import Message

    monkeypatch.setattr(__import__(__name__, fromlist=["RESOLVED"]), "RESOLVED", replace(RESOLVED, native_multimodal=True))
    requests = []

    def response(request):
        requests.append([m.to_openai_message() for m in request.messages])
        return LLMResponse("done")

    agent = Agent("parity", "Be concise.")
    config = RunConfig(
        workspace=tmp_path, initial_messages=[Message("user", "initial image", image_url="https://example.invalid/initial.png")]
    )
    steps = [
        LLMResponse(
            "", [ToolCall("a", "read_image", {"path": image_path}, extra_content={"provider": {"signature": "thought"}})]
        ),
        response,
    ]
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)
    assert requests[0] == requests[1]
    assert requests[0][-1]["content"][-1]["image_url"]["url"].startswith(
        "https://example.invalid/tool.png" if image_path.startswith("https") else "data:image/png;base64,"
    )


@pytest.mark.parametrize("cycles", [1, 2])
def test_continue_max_cycles_parity(tmp_path, cycles):
    agent = Agent("parity", "Be concise.", no_tool_policy="continue", max_cycles=cycles)
    steps = [LLMResponse(f"answer-{i}") for i in range(cycles)]
    config = RunConfig(workspace=tmp_path)
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)


def test_provider_default_settings_parity(tmp_path):
    def response(request):
        assert request.model_settings.temperature == 0.4
        assert request.model_settings.max_tokens == 120
        return LLMResponse("done")

    agent = Agent("parity", "Be concise.")
    config = RunConfig(workspace=tmp_path, model_settings=ModelSettings(max_tokens=120))
    defaults = ModelSettings(temperature=0.4, max_tokens=50)
    assert observe(agent, [response], config, kernel=True, provider_settings=defaults) == observe(
        agent,
        [response],
        config,
        kernel=False,
        provider_settings=defaults,
    )


def test_disabled_tool_never_executes(tmp_path):
    effects = []

    @function_tool(is_enabled=False)
    def disabled() -> str:
        effects.append("executed")
        return "unsafe"

    agent = Agent("parity", "Be concise.", tools=[disabled])
    steps = [LLMResponse("", [ToolCall("a", "disabled", {})]), LLMResponse("done")]
    config = RunConfig(workspace=tmp_path)
    assert observe(agent, steps, config, kernel=True) == observe(agent, steps, config, kernel=False)
    assert effects == []
