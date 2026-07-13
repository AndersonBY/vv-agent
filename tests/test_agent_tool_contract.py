from __future__ import annotations

from pathlib import Path

from vv_agent import Agent, ModelRef, ModelSettings, RunConfig, Runner, ScriptedModelProvider
from vv_agent.constants import TASK_FINISH_TOOL_NAME
from vv_agent.llm import LlmRequest
from vv_agent.types import LLMResponse, ToolCall


def _finish(message: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(id=f"finish-{message}", name=TASK_FINISH_TOOL_NAME, arguments={"message": message})],
    )


def test_agent_tool_schema_and_child_resolution_match_shared_contract(tmp_path: Path) -> None:
    requests: list[LlmRequest] = []
    responses = [
        LLMResponse(
            content="delegate",
            tool_calls=[
                ToolCall(
                    id="research",
                    name="research",
                    arguments={
                        "task_description": "find facts",
                        "output_requirements": "Return three bullets.",
                        "include_main_summary": True,
                    },
                )
            ],
        ),
        _finish("child facts"),
        _finish("parent final"),
    ]

    def respond(request: LlmRequest) -> LLMResponse:
        requests.append(request)
        return responses[len(requests) - 1]

    provider = ScriptedModelProvider.from_callback("scripted", "provider-model", respond)
    child = Agent(
        name="researcher",
        instructions="Research.",
        model=ModelRef.named("child-model"),
        model_settings=ModelSettings(temperature=0.7),
    )
    tool = child.as_tool(name="research", description="Research facts.")
    parent = Agent(
        name="writer",
        instructions="Delegate research.",
        model=ModelRef.named("parent-model"),
        model_settings=ModelSettings(temperature=0.1),
        tools=[tool],
    )

    result = Runner.run_sync(
        parent,
        "write report",
        run_config=RunConfig(
            model_provider=provider,
            model_settings=ModelSettings(temperature=0.2),
            workspace=tmp_path,
        ),
    )

    assert tool.params_json_schema == {
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "Task for the delegated agent.",
            },
            "output_requirements": {
                "type": "string",
                "description": "Optional output requirements for the delegated agent.",
            },
            "include_main_summary": {
                "type": "boolean",
                "description": "Whether to include parent task summary.",
            },
        },
        "required": ["task_description"],
        "additionalProperties": False,
    }
    assert result.final_output == "parent final"
    assert [request.model for request in requests] == ["parent-model", "child-model", "parent-model"]
    assert [request.model_settings.temperature for request in requests if request.model_settings is not None] == [
        0.2,
        0.7,
        0.2,
    ]
    child_prompt = next(message.content for message in requests[1].messages if message.role == "user")
    assert "<Output Requirements>\nReturn three bullets.\n</Output Requirements>" in child_prompt
    assert "<Main Task Summary>\nwrite report\n</Main Task Summary>" in child_prompt
