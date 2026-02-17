from __future__ import annotations

import json
from pathlib import Path

from v_agent.constants import TASK_FINISH_TOOL_NAME
from v_agent.llm import ScriptedLLM
from v_agent.runtime import AgentRuntime
from v_agent.runtime.tool_planner import plan_tool_schemas
from v_agent.tools import ToolContext, ToolSpec, build_default_registry
from v_agent.tools.registry import ToolRegistry
from v_agent.types import AgentStatus, AgentTask, LLMResponse, ToolCall, ToolExecutionResult

CUSTOM_WORKFLOW_TOOL_NAME = "_workflow_custom_run"


def _custom_workflow_tool(context: ToolContext, arguments: dict[str, object]) -> ToolExecutionResult:
    del context
    payload = {
        "ok": True,
        "workflow": arguments.get("workflow"),
        "message": "custom workflow executed",
    }
    return ToolExecutionResult(
        tool_call_id="",
        content=json.dumps(payload, ensure_ascii=False),
    )


def _register_custom_workflow_tool() -> ToolRegistry:
    registry = build_default_registry()
    registry.register_schema(
        CUSTOM_WORKFLOW_TOOL_NAME,
        {
            "type": "function",
            "function": {
                "name": CUSTOM_WORKFLOW_TOOL_NAME,
                "description": "Run workflow via custom integration layer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workflow": {"type": "string"},
                    },
                    "required": ["workflow"],
                },
            },
        },
    )
    registry.register(ToolSpec(name=CUSTOM_WORKFLOW_TOOL_NAME, handler=_custom_workflow_tool))
    return registry


def test_custom_tool_can_be_injected_via_extra_tool_names() -> None:
    registry = _register_custom_workflow_tool()
    task = AgentTask(
        task_id="custom_schema",
        model="m",
        system_prompt="sys",
        user_prompt="u",
        extra_tool_names=[CUSTOM_WORKFLOW_TOOL_NAME],
    )

    schemas = plan_tool_schemas(registry=registry, task=task)
    names = {schema["function"]["name"] for schema in schemas}
    assert CUSTOM_WORKFLOW_TOOL_NAME in names


def test_runtime_executes_custom_workflow_tool(tmp_path: Path) -> None:
    registry = _register_custom_workflow_tool()
    llm = ScriptedLLM(
        steps=[
            LLMResponse(
                content="run custom workflow",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name=CUSTOM_WORKFLOW_TOOL_NAME,
                        arguments={"workflow": "wf_translate"},
                    )
                ],
            ),
            LLMResponse(
                content="finish",
                tool_calls=[ToolCall(id="c2", name=TASK_FINISH_TOOL_NAME, arguments={"message": "done"})],
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=llm, tool_registry=registry, default_workspace=tmp_path)
    task = AgentTask(
        task_id="custom_runtime",
        model="m",
        system_prompt="sys",
        user_prompt="u",
        extra_tool_names=[CUSTOM_WORKFLOW_TOOL_NAME],
        max_cycles=4,
    )

    result = runtime.run(task)
    assert result.status == AgentStatus.COMPLETED
    assert result.final_answer == "done"
    assert len(result.cycles) == 2
