from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from v_agent.runtime.hooks import RuntimeHookManager
from v_agent.tools import ToolContext, ToolRegistry
from v_agent.tools.dispatcher import dispatch_tool_call
from v_agent.types import AgentTask, CycleRecord, Message, ToolCall, ToolDirective, ToolExecutionResult, ToolResultStatus


@dataclass(slots=True)
class ToolRunOutcome:
    directive_result: ToolExecutionResult | None = None
    interruption_messages: list[Message] = field(default_factory=list)


class ToolCallRunner:
    def __init__(self, *, tool_registry: ToolRegistry, hook_manager: RuntimeHookManager | None = None) -> None:
        self.tool_registry = tool_registry
        self.hook_manager = hook_manager or RuntimeHookManager()

    def run(
        self,
        *,
        task: AgentTask,
        tool_calls: list[ToolCall],
        context: ToolContext,
        messages: list[Message],
        cycle_record: CycleRecord,
        interruption_provider: Callable[[], list[Message]] | None = None,
        on_tool_result: Callable[[ToolCall, ToolExecutionResult], None] | None = None,
    ) -> ToolRunOutcome:
        latest_directive_result: ToolExecutionResult | None = None
        interruption_messages: list[Message] = []

        for index, call in enumerate(tool_calls):
            patched_call, short_circuit_result = self.hook_manager.apply_before_tool_call(
                task=task,
                cycle_index=context.cycle_index,
                call=call,
                context=context,
            )
            if short_circuit_result is not None:
                result = short_circuit_result
                if not result.tool_call_id:
                    result.tool_call_id = call.id
            else:
                result = dispatch_tool_call(
                    registry=self.tool_registry,
                    context=context,
                    call=patched_call,
                )
            result = self.hook_manager.apply_after_tool_call(
                task=task,
                cycle_index=context.cycle_index,
                call=patched_call,
                context=context,
                result=result,
            )

            cycle_record.tool_results.append(result)
            messages.append(result.to_tool_message())
            self._append_image_notification(result=result, messages=messages)
            if on_tool_result is not None:
                on_tool_result(call, result)

            if result.directive in (ToolDirective.WAIT_USER, ToolDirective.FINISH):
                latest_directive_result = result
                break

            if interruption_provider is not None:
                pending_messages = interruption_provider()
                if pending_messages:
                    interruption_messages.extend(pending_messages)
                    for skipped_call in tool_calls[index + 1 :]:
                        skipped = self._build_skipped_result(skipped_call)
                        cycle_record.tool_results.append(skipped)
                        messages.append(skipped.to_tool_message())
                        if on_tool_result is not None:
                            on_tool_result(skipped_call, skipped)
                    break

        return ToolRunOutcome(
            directive_result=latest_directive_result,
            interruption_messages=interruption_messages,
        )

    @staticmethod
    def _append_image_notification(*, result: ToolExecutionResult, messages: list[Message]) -> None:
        if result.image_url:
            image_ref = result.image_path or result.image_url
            messages.append(
                Message(
                    role="user",
                    content=f"[Image loaded] {image_ref}",
                    image_url=result.image_url,
                )
            )
        elif result.image_path:
            messages.append(Message(role="user", content=f"[Image loaded] {result.image_path}"))

    @staticmethod
    def _build_skipped_result(call: ToolCall) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_call_id=call.id,
            status="error",
            status_code=ToolResultStatus.ERROR,
            error_code="skipped_due_to_steering",
            content=json.dumps(
                {
                    "ok": False,
                    "error": "Tool skipped due to queued steering message.",
                    "error_code": "skipped_due_to_steering",
                },
                ensure_ascii=False,
            ),
        )
