from __future__ import annotations

from v_agent.tools import ToolContext, ToolRegistry
from v_agent.tools.dispatcher import dispatch_tool_call
from v_agent.types import CycleRecord, Message, ToolCall, ToolDirective, ToolExecutionResult


class ToolCallRunner:
    def __init__(self, *, tool_registry: ToolRegistry) -> None:
        self.tool_registry = tool_registry

    def run(
        self,
        *,
        tool_calls: list[ToolCall],
        context: ToolContext,
        messages: list[Message],
        cycle_record: CycleRecord,
    ) -> ToolExecutionResult | None:
        latest_directive_result: ToolExecutionResult | None = None

        for call in tool_calls:
            result = dispatch_tool_call(
                registry=self.tool_registry,
                context=context,
                call=call,
            )

            cycle_record.tool_results.append(result)
            messages.append(result.to_tool_message())
            self._append_image_notification(result=result, messages=messages)

            if result.directive in (ToolDirective.WAIT_USER, ToolDirective.FINISH):
                latest_directive_result = result
                break

        return latest_directive_result

    @staticmethod
    def _append_image_notification(*, result: ToolExecutionResult, messages: list[Message]) -> None:
        if result.image_url:
            messages.append(Message(role="user", content=f"[Image loaded] {result.image_url}"))
        elif result.image_path:
            messages.append(Message(role="user", content=f"[Image loaded] {result.image_path}"))
