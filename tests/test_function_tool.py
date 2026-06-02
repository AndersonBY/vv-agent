from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from vv_agent import ToolContext, ToolOutputJson, ToolOutputText, function_tool
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

    output = lookup_order.on_invoke(None, {"order_id": "A-1", "include_history": True})

    assert output == ToolOutputText(text="A-1:True:10")


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
