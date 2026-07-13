from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TypedDict

from vv_agent import (
    FunctionTool,
    ToolContext,
    ToolOutputError,
    ToolOutputFile,
    ToolOutputImage,
    ToolOutputJson,
    ToolOutputText,
    function_tool,
)
from vv_agent.tools import ToolExposure
from vv_agent.types import ToolCall
from vv_agent.workspace import MemoryWorkspaceBackend


def test_function_tool_infers_schema_from_signature_and_docstring() -> None:
    @function_tool
    def lookup_order(order_id: str, include_history: bool = False, limit: int = 10) -> str:
        """Lookup an order."""
        return f"{order_id}:{include_history}:{limit}"

    assert lookup_order.name == "lookup_order"
    assert lookup_order.description == "Lookup an order."
    assert lookup_order.params_json_schema == {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "include_history": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["order_id"],
        "additionalProperties": False,
    }
    assert lookup_order.to_openai_schema()["function"]["strict"] is True

    output = lookup_order.on_invoke(None, {"order_id": "A-1", "include_history": True})

    assert output == ToolOutputText(text="A-1:True:10")


def test_function_tool_schema_strictness_and_exposure_are_model_visible_contracts() -> None:
    @function_tool(strict_json_schema=False, exposure=ToolExposure.HIDDEN)
    def hidden(value: str) -> str:
        return value

    assert hidden.strict_json_schema is False
    assert hidden.exposure == ToolExposure.HIDDEN
    assert hidden.to_executor().exposure == ToolExposure.HIDDEN
    assert hidden.to_openai_schema()["function"]["strict"] is False


def test_function_tool_timeout_returns_structured_retryable_error(tmp_path) -> None:
    @function_tool(timeout_seconds=0.01)
    def slow() -> str:
        time.sleep(0.05)
        return "late"

    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=0,
        workspace_backend=MemoryWorkspaceBackend(),
    )
    result = slow.to_executor().execute(ToolCall(id="slow-call", name="slow", arguments={}), context)

    assert result.error_code == "tool_timeout"
    assert json.loads(result.content)["retryable"] is True


def test_manual_function_tool_normalizes_formatted_and_default_failures(tmp_path) -> None:
    def fail(_context, _arguments):
        raise RuntimeError("database unavailable")

    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=0,
        workspace_backend=MemoryWorkspaceBackend(),
    )
    formatted = FunctionTool(
        name="formatted",
        description="Fail with a host-formatted error.",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke=fail,
        failure_error_function=lambda exc: f"host error: {exc}",
    )
    default = FunctionTool(
        name="default",
        description="Fail with the framework default error.",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke=fail,
    )

    formatted_result = formatted.to_executor().execute(
        ToolCall(id="formatted-call", name="formatted", arguments={}),
        context,
    )
    default_result = default.to_executor().execute(
        ToolCall(id="default-call", name="default", arguments={}),
        context,
    )

    assert formatted_result.error_code == "tool_execution_failed"
    assert json.loads(formatted_result.content)["error"] == "host error: database unavailable"
    assert default_result.error_code == "tool_execution_failed"
    assert "Tool execution failed (default): database unavailable" in default_result.content


def test_function_tool_infers_schema_from_dataclass_argument() -> None:
    @dataclass
    class SearchInput:
        pattern: str
        max_results: int = 20

    @function_tool
    def search_workspace(args: SearchInput) -> dict[str, object]:
        """Search workspace."""
        return {"pattern": args.pattern, "max_results": args.max_results}

    assert search_workspace.params_json_schema == {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "max_results": {"type": "integer", "default": 20},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    placeholder_text = "".join(("TO", "DO"))
    assert search_workspace.on_invoke(None, {"pattern": placeholder_text}) == ToolOutputJson(
        data={"pattern": placeholder_text, "max_results": 20}
    )


def test_function_tool_infers_schema_from_typed_dict_argument() -> None:
    class ReviewInput(TypedDict):
        path: str
        strict: bool

    @function_tool(name="review_file", description="Review a file.")
    def review(args: ReviewInput) -> str:
        return f"{args['path']}:{args['strict']}"

    assert review.name == "review_file"
    assert review.description == "Review a file."
    assert review.params_json_schema == {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "strict": {"type": "boolean"},
        },
        "required": ["path", "strict"],
        "additionalProperties": False,
    }
    assert review.on_invoke(None, {"path": "a.py", "strict": True}) == ToolOutputText(text="a.py:True")


def test_function_tool_excludes_tool_context_from_schema_and_passes_it(tmp_path) -> None:
    @function_tool
    def inspect_context(context: ToolContext, value: str) -> str:
        """Inspect tool context."""
        return f"{context.tool_name}:{context.tool_call_id}:{context.arguments['value']}:{value}"

    assert inspect_context.params_json_schema == {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    context = ToolContext(
        workspace=tmp_path,
        shared_state={},
        cycle_index=0,
        workspace_backend=MemoryWorkspaceBackend(),
        tool_call_id="call-1",
        tool_name="inspect_context",
        arguments={"value": "payload"},
    )

    assert inspect_context.on_invoke(context, {"value": "payload"}) == ToolOutputText(
        text="inspect_context:call-1:payload:payload"
    )


def test_tool_output_variants_preserve_shared_metadata_contract() -> None:
    @function_tool
    def noop() -> str:
        return "ok"

    json_result = noop.to_tool_execution_result(
        ToolOutputJson(data={"ok": True}, metadata={"source": "test"}), tool_call_id="json"
    )
    assert json_result.metadata == {"output_type": "json", "source": "test"}

    image_result = noop.to_tool_execution_result(
        ToolOutputImage(
            url="https://example.invalid/image.png",
            path="image.png",
            mime_type="image/png",
            metadata={"source": "test"},
        ),
        tool_call_id="image",
    )
    assert json.loads(image_result.content)["mime_type"] == "image/png"
    assert image_result.metadata == {"output_type": "image", "source": "test"}

    file_result = noop.to_tool_execution_result(
        ToolOutputFile(path="report.json", mime_type="application/json", metadata={"source": "test"}),
        tool_call_id="file",
    )
    assert file_result.metadata == {"output_type": "file", "source": "test"}

    error_result = noop.to_tool_execution_result(
        ToolOutputError(
            message="temporary failure",
            error_code="temporary",
            retryable=True,
            metadata={"source": "test"},
        ),
        tool_call_id="error",
    )
    assert json.loads(error_result.content) == {
        "ok": False,
        "error": "temporary failure",
        "error_code": "temporary",
        "retryable": True,
    }
    assert error_result.metadata == {"output_type": "error", "retryable": True, "source": "test"}
