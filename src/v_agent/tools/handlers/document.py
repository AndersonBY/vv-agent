from __future__ import annotations

from typing import Any

from v_agent.tools.base import ToolContext
from v_agent.tools.handlers.common import to_json
from v_agent.types import ToolExecutionResult, ToolResultStatus


def _not_enabled(tool_name: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_call_id="",
        status="error",
        status_code=ToolResultStatus.ERROR,
        error_code="document_tools_not_enabled",
        content=to_json(
            {
                "error": f"{tool_name} is not enabled in this runtime",
                "error_code": "document_tools_not_enabled",
            }
        ),
    )


def document_list_mounted(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("document_list_mounted")


def document_read(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("document_read")


def document_grep(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("document_grep")


def document_abstract_read(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("document_abstract_read")


def document_overview_read(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("document_overview_read")


def folder_abstract_read(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("folder_abstract_read")


def document_find(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("document_find")


def document_write(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("document_write")


def document_str_replace(context: ToolContext, arguments: dict[str, Any]) -> ToolExecutionResult:
    del context, arguments
    return _not_enabled("document_str_replace")
