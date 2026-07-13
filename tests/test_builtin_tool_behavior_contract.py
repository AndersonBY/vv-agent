from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vv_agent.constants import (
    ACTIVATE_SKILL_TOOL_NAME,
    ASK_USER_TOOL_NAME,
    BASH_TOOL_NAME,
    CHECK_BACKGROUND_COMMAND_TOOL_NAME,
    COMPRESS_MEMORY_TOOL_NAME,
    FILE_INFO_TOOL_NAME,
    FIND_FILES_TOOL_NAME,
    READ_IMAGE_TOOL_NAME,
    TASK_FINISH_TOOL_NAME,
)
from vv_agent.prompt import build_system_prompt
from vv_agent.runtime.background_sessions import background_session_manager
from vv_agent.runtime.tool_planner import plan_tool_schemas
from vv_agent.tools import RegistryToolExecutor, ToolContext, ToolExposure, build_default_registry
from vv_agent.types import AgentTask, ToolCall, ToolExecutionResult
from vv_agent.workspace import LocalWorkspaceBackend

_FIXTURE_PATH = Path(__file__).parent / "fixtures/parity/builtin_tool_behavior_v1.json"
_FIXTURE: dict[str, Any] = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace=tmp_path,
        shared_state={"todo_list": []},
        cycle_index=1,
        workspace_backend=LocalWorkspaceBackend(tmp_path),
    )


def _assert_result(result: ToolExecutionResult, expected: dict[str, Any]) -> None:
    wire = result.to_dict()
    for key in ("status", "status_code", "directive"):
        assert wire[key] == expected[key]
    assert wire.get("error_code") == expected.get("error_code")
    assert json.loads(result.content) == expected["content"]
    assert result.metadata == expected["metadata"]

    if expected["status"] == "error":
        required = set(_FIXTURE["canonical"]["error_content_required_keys"])
        assert required <= set(json.loads(result.content))

    forbidden = set(_FIXTURE["canonical"]["metadata_policy"]["forbidden_large_keys"])
    assert forbidden.isdisjoint(result.metadata)


def _execute(
    registry: Any,
    context: ToolContext,
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolExecutionResult:
    return registry.execute(
        ToolCall(id=f"fixture-{tool_name}", name=tool_name, arguments=arguments),
        context,
    )


def test_fixture_drives_prompt_registry_dynamic_hint_and_projection(tmp_path: Path) -> None:
    empty_skills = _FIXTURE["prompt"]["empty_skills"]
    prompt = build_system_prompt(
        "Fixture agent",
        available_skills=empty_skills["available_skills"],
    )
    for fragment in empty_skills["forbidden_fragments"]:
        assert fragment not in prompt

    registry = build_default_registry()
    description_case = _FIXTURE["registry"]["builtin_description"]
    executor = registry.get_executor(description_case["tool_name"])
    assert bool(executor.description.strip()) is description_case["must_be_non_empty"]
    assert executor.description == registry.get_schema(description_case["tool_name"])["function"]["description"]

    hidden_case = _FIXTURE["registry"]["hidden_exposure"]

    def hidden_handler(_context: ToolContext, _arguments: dict[str, Any]) -> ToolExecutionResult:
        return ToolExecutionResult(tool_call_id="", content="{}")

    registry.register_executor(
        RegistryToolExecutor(
            name=hidden_case["tool_name"],
            description="Fixture hidden tool",
            handler=hidden_handler,
            exposure=ToolExposure.HIDDEN,
        )
    )
    visible_names = {schema["function"]["name"] for schema in registry.list_openai_schemas()}
    assert (hidden_case["tool_name"] in visible_names) is hidden_case["must_be_model_visible"]

    hint_case = _FIXTURE["dynamic_bash_description"]["non_string_bash_shell"]
    task = AgentTask(
        task_id="fixture-task",
        model="fixture-model",
        system_prompt="system",
        user_prompt="user",
        agent_type="computer",
        metadata={"bash_shell": hint_case["bash_shell"]},
    )
    schemas = plan_tool_schemas(registry=registry, task=task)
    bash_schema = next(schema for schema in schemas if schema["function"]["name"] == BASH_TOOL_NAME)
    assert hint_case["expected_hint"] in bash_schema["function"]["description"]

    projection = _FIXTURE["tool_execution_result_projection"]
    projected = ToolExecutionResult.from_dict(projection["canonical"])
    assert projected.to_dict() == projection["canonical"]
    for field in projection["forbidden_absent_fields"]:
        assert field not in projected.to_dict()


def test_fixture_drives_builtin_handler_envelopes_and_metadata(tmp_path: Path) -> None:
    registry = build_default_registry()
    tools = _FIXTURE["tools"]

    context = _context(tmp_path)
    case = tools["compress_memory"]["success"]
    _assert_result(
        _execute(registry, context, COMPRESS_MEMORY_TOOL_NAME, case["arguments"]),
        case["result"],
    )
    case = tools["compress_memory"]["missing_core_information"]
    _assert_result(
        _execute(registry, context, COMPRESS_MEMORY_TOOL_NAME, case["arguments"]),
        case["result"],
    )

    skill_case = tools["activate_skill"]["success"]
    context = _context(tmp_path)
    context.shared_state["available_skills"] = skill_case["available_skills"]
    _assert_result(
        _execute(registry, context, ACTIVATE_SKILL_TOOL_NAME, skill_case["arguments"]),
        skill_case["result"],
    )

    context = _context(tmp_path)
    context.metadata["available_skills"] = skill_case["available_skills"]
    source_case = tools["activate_skill"]["metadata_only_source"]
    _assert_result(
        _execute(registry, context, ACTIVATE_SKILL_TOOL_NAME, skill_case["arguments"]),
        source_case["result"],
    )

    image_case = tools["read_image"]["too_large"]
    (tmp_path / image_case["path"]).write_bytes(b"x" * image_case["actual_bytes"])
    _assert_result(
        _execute(registry, _context(tmp_path), READ_IMAGE_TOOL_NAME, {"path": image_case["path"]}),
        image_case["result"],
    )

    file_info_case = tools["file_info"]["missing_path"]
    _assert_result(
        _execute(registry, _context(tmp_path), FILE_INFO_TOOL_NAME, file_info_case["arguments"]),
        file_info_case["result"],
    )
    find_files_case = tools["find_files"]["missing_directory"]
    _assert_result(
        _execute(registry, _context(tmp_path), FIND_FILES_TOOL_NAME, find_files_case["arguments"]),
        find_files_case["result"],
    )

    finish_case = tools["control"]["blank_task_finish"]
    _assert_result(
        _execute(registry, _context(tmp_path), TASK_FINISH_TOOL_NAME, finish_case["arguments"]),
        finish_case["result"],
    )
    ask_case = tools["control"]["blank_ask_user"]
    _assert_result(
        _execute(registry, _context(tmp_path), ASK_USER_TOOL_NAME, ask_case["arguments"]),
        ask_case["result"],
    )


def test_fixture_drives_bash_and_background_command_contract(tmp_path: Path) -> None:
    registry = build_default_registry()
    tools = _FIXTURE["tools"]

    non_zero = tools["bash"]["non_zero"]
    context = _context(tmp_path)
    context.task_metadata = dict(non_zero["context_metadata"])
    _assert_result(
        _execute(registry, context, BASH_TOOL_NAME, non_zero["arguments"]),
        non_zero["result"],
    )

    invalid_timeout = tools["bash"]["invalid_timeout"]
    _assert_result(
        _execute(registry, _context(tmp_path), BASH_TOOL_NAME, invalid_timeout["arguments"]),
        invalid_timeout["result"],
    )

    background = tools["bash"]["background_start"]
    context = _context(tmp_path)
    context.task_metadata = dict(background["context_metadata"])
    result = _execute(registry, context, BASH_TOOL_NAME, background["arguments"])
    wire = result.to_dict()
    expected = background["result"]
    for key in ("status", "status_code", "directive"):
        assert wire[key] == expected[key]
    content = json.loads(result.content)
    for key, value in expected["content_subset"].items():
        assert content[key] == value
    for key, value in expected["metadata_subset"].items():
        assert result.metadata[key] == value
    for key in expected["required_dynamic_fields"]:
        assert content[key]
        assert result.metadata[key] == content[key]

    forbidden = set(_FIXTURE["canonical"]["metadata_policy"]["forbidden_large_keys"])
    assert forbidden.isdisjoint(result.metadata)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if background_session_manager.check(content["session_id"])["status"] != "running":
            break
        time.sleep(0.02)

    missing = tools["check_background_command"]["missing_session"]
    _assert_result(
        _execute(registry, _context(tmp_path), CHECK_BACKGROUND_COMMAND_TOOL_NAME, missing["arguments"]),
        missing["result"],
    )
