from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from support import ModelMapProvider
from test_agent_as_tool import _resolved

from vv_agent import Agent, ApprovalDecision, RunConfig, Runner, ToolPolicy, function_tool
from vv_agent.app_server import AppServer, ChannelTransport
from vv_agent.app_server.host import DefaultAppServerHost
from vv_agent.interactive import AgentSessionOptions, InteractiveAgentClient, InteractiveAgentDefinition
from vv_agent.llm import ScriptedLLM
from vv_agent.runtime.cancellation import CancellationToken, CancelledError
from vv_agent.tools import ToolContext, build_default_registry
from vv_agent.tools.function import FunctionTool
from vv_agent.tools.orchestrator import ToolOrchestrator
from vv_agent.types import LLMResponse, SubAgentConfig, ToolCall

ENTRYPOINTS = [
    "invoke",
    "executor",
    "orchestrator",
    "registry",
    "runner_executor",
    "interactive",
    "configured_child",
    "app_server",
    "background",
    "runner",
    "runtime",
    "runtime_configured_child",
]


def _run_parent(entrypoint, child, config, parent_llm):
    tool = child.as_tool()
    call = ToolCall(id="delegate", name=tool.name, arguments={"task_description": "Do child work"})
    parent_llm.steps[:] = [LLMResponse(content="", tool_calls=[call]), LLMResponse(content="parent done")]
    parent = Agent(name="parent", instructions="Delegate.", model="parent")
    registry = build_default_registry()
    registry.register_executor(tool.to_executor())
    if entrypoint in {"invoke", "executor", "orchestrator"}:

        def invoke(context, arguments):
            if entrypoint == "invoke":
                return tool.invoke(context, arguments)
            if entrypoint == "executor":
                return tool.to_executor().execute(call, context)
            return ToolOrchestrator.from_tools([tool]).run_one(call, context=context)

        parent.tools = [
            FunctionTool(
                name=tool.name, description=tool.description, params_json_schema=tool.params_json_schema, on_invoke=invoke
            )
        ]
    elif entrypoint == "runner":
        parent.tools = [tool]
    elif entrypoint in {"runner_executor", "background"}:
        parent.tools = [tool.to_executor()]
    else:
        config = replace(config, tool_registry_factory=lambda: registry)

    if entrypoint == "interactive":
        client = InteractiveAgentClient(
            options=AgentSessionOptions(
                model_provider=config.model_provider,
                workspace=config.workspace,
                tool_registry_factory=config.tool_registry_factory,
                cancellation_token=config.cancellation_token,
                tool_policy=config.tool_policy,
                approval_provider=config.approval_provider,
            )
        )
        session = client.create_session(agent=InteractiveAgentDefinition(description="Delegate.", model="parent"))
        try:
            session.prompt("go")
        finally:
            session.close()
    elif entrypoint in {"configured_child", "runtime_configured_child"}:
        config.model_provider.routes["middle"] = (ScriptedLLM(steps=list(parent_llm.steps)), _resolved("middle"))
        parent.sub_agents = {"middle": SubAgentConfig(model="middle", description="Delegate.")}
        parent_llm.steps[:] = [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="configured",
                        name="create_sub_task",
                        arguments={"agent_id": "middle", "task_description": "Delegate", "wait_for_completion": True},
                    )
                ],
            ),
            LLMResponse(content="parent done"),
        ]
        if entrypoint == "runtime_configured_child":
            _run_runtime(parent, config, parent_llm, registry)
        else:
            Runner.run_sync(parent, "go", run_config=config)
    elif entrypoint == "runtime":
        _run_runtime(parent, config, parent_llm, registry)
    elif entrypoint == "app_server":
        transport = ChannelTransport(connection_id="scope-test")
        server = AppServer(transport=transport, host=DefaultAppServerHost(agent=parent, run_config=config))
        for payload in [
            {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"clientInfo": {"name": "test"}}},
            {"jsonrpc": "2.0", "method": "initialized"},
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "thread/start",
                "params": {"agentKey": "default", "cwd": str(config.workspace)},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "turn/start",
                "params": {"threadId": "thread_1", "input": [{"type": "text", "text": "go"}]},
            },
        ]:
            transport.send_inbound(payload)
            server.processor.process_next(transport)
        while transport.receive_outbound(timeout=10).get("method") != "turn/completed":
            pass
    elif entrypoint == "background":
        parent.as_background_task().start(Runner, None, {"task_description": "go"}, run_config=config).wait(timeout=10)
    else:
        Runner.run_sync(parent, "go", run_config=config)


def _run_runtime(parent, config, llm, registry):
    from vv_agent.prompt import build_raw_system_prompt_bundle
    from vv_agent.runtime import AgentRuntime, ExecutionContext
    from vv_agent.types import AgentTask

    task = AgentTask(
        task_id="raw-parent",
        model="parent",
        user_prompt="go",
        prompt_bundle=build_raw_system_prompt_bundle("Delegate."),
        extra_tool_names=["child"],
        sub_agents=parent.sub_agents,
    )
    AgentRuntime(
        llm_client=llm,
        model_provider=config.model_provider,
        tool_registry=registry,
        default_workspace=config.workspace,
    ).run(
        task,
        ctx=ExecutionContext(
            cancellation_token=config.cancellation_token,
            metadata={
                "_vv_agent_approval_provider": config.approval_provider,
                "_vv_agent_denied_side_effects": config.tool_policy.denied_side_effects if config.tool_policy else [],
            },
        ),
    )


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("capability", ["provider", "cancellation", "policy", "approval"])
def test_agent_tool_inherits_parent_scope(entrypoint, capability, tmp_path: Path, monkeypatch):
    token = CancellationToken()
    effects = []
    observations = []
    approval_requests = []

    class DenyApproval:
        def should_request(self, request):
            return request.tool_name == "forbidden"

        def decide(self, request):
            approval_requests.append(request)
            return ApprovalDecision.deny("parent denial")

    @function_tool(needs_approval=capability == "approval", tool_metadata={"side_effect": "write"})
    def forbidden() -> str:
        effects.append("executed")
        return "effect"

    def child_step(request):
        observations.append(request)
        if capability == "cancellation":
            token.cancel("parent cancelled during child model call")
        return LLMResponse(content="", tool_calls=[ToolCall(id="effect", name="forbidden", arguments={})])

    child_llm = ScriptedLLM(steps=[child_step, LLMResponse(content="child done")])
    parent_llm = ScriptedLLM()
    provider = ModelMapProvider(
        routes={
            "parent": (parent_llm, _resolved("parent")),
            "child": (child_llm, _resolved("child")),
        },
        default_model="parent",
    )
    config = RunConfig(
        workspace=tmp_path,
        model_provider=provider,
        cancellation_token=token,
        tool_policy=ToolPolicy(denied_side_effects=["write"]) if capability == "policy" else None,
    )
    if capability == "approval":
        config.approval_provider = DenyApproval()
    child = Agent(name="child", instructions="Do work.", model="child", tools=[forbidden])

    if capability != "provider":
        # The broken closure first fails for lack of a provider. Supply only that
        # capability at the child boundary to expose the independent scope losses.
        run_sync = Runner.run_sync

        def run_with_test_provider(cls, agent, input, *, run_config=None):
            if agent is child:
                run_config = replace(run_config or RunConfig(), model_provider=provider)
            return run_sync(agent, input, run_config=run_config)

        monkeypatch.setattr(Runner, "run_sync", classmethod(run_with_test_provider))

    try:
        _run_parent(entrypoint, child, config, parent_llm)
    except CancelledError:
        assert capability == "cancellation"
    assert observations, "child did not use the parent model provider"
    if capability == "provider":
        assert "child" in provider.resolved_models
    else:
        assert effects == [], f"parent {capability} failed to prevent the child effect"
        if capability == "approval":
            assert len(approval_requests) == 1
        if capability == "cancellation":
            assert len(child_llm.steps) == 1, "cancelled child continued to a second model call"


def test_agent_tool_without_scope_fails_closed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(Runner, "run_sync", lambda *args, **kwargs: calls.append(kwargs))
    tool = Agent(name="child", instructions="Do work.", model="child").as_tool()
    result = tool.to_tool_execution_result(tool.invoke(None, {"task_description": "go"}))
    assert calls == []
    assert result.error_code == "sub_agents_not_enabled"


def test_agent_tool_preserves_runtime_capabilities_and_child_isolation(tmp_path):
    from vv_agent.budget import RunBudgetLimits
    from vv_agent.runtime.backends import InlineBackend
    from vv_agent.workspace import MemoryWorkspaceBackend

    captured = []

    @function_tool
    def inspect_scope(context: ToolContext) -> str:
        captured.append(context)
        context.shared_state["child_only"] = True
        return "captured"

    child_llm = ScriptedLLM(
        steps=[
            LLMResponse(content="", tool_calls=[ToolCall(id="inspect", name="inspect_scope", arguments={})]),
            LLMResponse(content="child done"),
        ]
    )
    parent_llm = ScriptedLLM()
    provider = ModelMapProvider(
        routes={
            "parent": (parent_llm, _resolved("parent")),
            "child": (child_llm, _resolved("child")),
        },
        default_model="parent",
    )
    workspace = MemoryWorkspaceBackend()
    backend = InlineBackend()
    token = CancellationToken()
    state = {"parent_value": [1]}
    app_state = object()
    limits = RunBudgetLimits(max_tool_calls=4)
    config = RunConfig(
        workspace=tmp_path,
        workspace_backend=workspace,
        execution_backend=backend,
        model_provider=provider,
        cancellation_token=token,
        budget_limits=limits,
        metadata={"tenant": "parent"},
        shared_state=state,
        context=app_state,
    )
    child = Agent(name="child", instructions="Inspect.", model="child", tools=[inspect_scope])
    _run_parent("executor", child, config, parent_llm)
    assert len(captured) == 1
    context = captured[0]
    assert context.workspace == tmp_path
    assert context.workspace_backend is workspace
    assert context.run_context.context is app_state
    assert context.task_metadata["tenant"] == "parent"
    assert context.ctx.metadata["execution_backend"] is backend
    assert context.ctx.metadata["_vv_agent_budget_limits"] == limits
    assert context.ctx.metadata["_vv_agent_model_provider"] is provider
    assert state == {"parent_value": [1]}
    child_token = context.ctx.cancellation_token
    assert child_token is not token
    child_token.cancel("child only")
    assert not token.cancelled
